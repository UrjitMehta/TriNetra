from imports import *

#=========================== Transformer(SwinUnet) + Conditioned Attention + Boundary + Skip ===========================
# ==========================================================
# Utility Layers
# ==========================================================

@keras.saving.register_keras_serializable()
class InputGuidance(layers.Layer):
    """Extract the last channel as guidance."""
    def call(self, x):
        return x[..., -1:]

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], 1)


@keras.saving.register_keras_serializable()
class ResizeToMatch(layers.Layer):
    """Resize the first tensor to match the spatial dimensions of the second tensor."""
    def call(self, inputs):
        to_resize, reference = inputs
        return tf.image.resize(
            to_resize,
            (tf.shape(reference)[1], tf.shape(reference)[2]),
            method='bilinear'
        )


@keras.saving.register_keras_serializable()
class CastToDtypeLike(layers.Layer):
    """Cast a tensor to the dtype of a reference tensor."""
    def call(self, inputs):
        x, like = inputs
        return tf.cast(x, like.dtype)


# ==========================================================
# MLP + Bias + Window Utilities
# ==========================================================

@keras.saving.register_keras_serializable()
class MLPBlock(layers.Layer):
    def __init__(self, hidden_dim, out_dim, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim, self.out_dim = hidden_dim, out_dim

    def build(self, _):
        self.dense1 = layers.Dense(
            self.hidden_dim,
            activation='gelu',
            dtype='float32'
        )
        self.dense2 = layers.Dense(
            self.out_dim,
            dtype='float32'
        )

    def call(self, x):
        x = tf.cast(x, tf.float32)
        return self.dense2(self.dense1(x))


@keras.saving.register_keras_serializable()
class AdditiveSpatialBias(layers.Layer):
    def __init__(self, num_heads, window_size, **kwargs):
        super().__init__(**kwargs)
        self.num_heads, self.window_size = num_heads, window_size

    def build(self, _):
        self.learned_bias = self.add_weight(
            shape=(
                self.num_heads,
                self.window_size * self.window_size,
                self.window_size * self.window_size
            ),
            initializer='zeros',
            trainable=True,
            name='spatial_bias'
        )

        self.guidance_scale = self.add_weight(
            shape=(self.num_heads, 1, 1),
            initializer='zeros',
            trainable=True,
            name='guidance_scale'
        )

    def call(self, attn_scores, guide_vec_windows=None):
        bias = tf.cast(
            self.learned_bias[tf.newaxis, :, :, :],
            attn_scores.dtype
        )

        if guide_vec_windows is None:
            return attn_scores + bias

        g = tf.reshape(
            guide_vec_windows,
            [tf.shape(guide_vec_windows)[0], tf.shape(guide_vec_windows)[1]]
        )

        g_i, g_j = tf.expand_dims(g, 2), tf.expand_dims(g, 1)
        g_pair = tf.expand_dims(0.5 * (g_i + g_j), 1)

        g_pair *= tf.cast(
            self.guidance_scale,
            attn_scores.dtype
        )[tf.newaxis, :, :, :]

        return attn_scores + bias + g_pair


def window_partition(x, window_size):
    B, H, W, C = (
        tf.shape(x)[0],
        tf.shape(x)[1],
        tf.shape(x)[2],
        tf.shape(x)[3]
    )

    pad_h = (window_size - (H % window_size)) % window_size
    pad_w = (window_size - (W % window_size)) % window_size

    x = tf.pad(
        x,
        [[0, 0], [0, pad_h], [0, pad_w], [0, 0]]
    )

    Hp, Wp = H + pad_h, W + pad_w

    x = tf.reshape(
        x,
        [
            B,
            Hp // window_size,
            window_size,
            Wp // window_size,
            window_size,
            C
        ]
    )

    x = tf.transpose(x, [0, 1, 3, 2, 4, 5])

    return (
        tf.reshape(
            x,
            [
                B * (Hp // window_size) * (Wp // window_size),
                window_size * window_size,
                C
            ]
        ),
        (H, W, Hp, Wp, pad_h, pad_w)
    )


def window_reverse(windows, window_size, meta):
    H, W, Hp, Wp, pad_h, pad_w = meta

    C = tf.shape(windows)[-1]
    Nh, Nw = Hp // window_size, Wp // window_size
    B = tf.shape(windows)[0] // (Nh * Nw)

    x = tf.reshape(
        windows,
        [B, Nh, Nw, window_size, window_size, C]
    )

    x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
    x = tf.reshape(x, [B, Hp, Wp, C])

    return x[:, :H, :W, :]


# ==========================================================
# Swin Transformer Block with Attention Guidance
# ==========================================================

@keras.saving.register_keras_serializable()
class SwinTransformerBlock(layers.Layer):
    def __init__(
        self,
        num_heads=4,
        mlp_ratio=4.,
        window_size=7,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_heads, self.mlp_ratio, self.window_size = (
            num_heads,
            mlp_ratio,
            window_size
        )

        self.norm1 = layers.LayerNormalization(
            epsilon=1e-5,
            dtype='float32'
        )
        self.norm2 = layers.LayerNormalization(
            epsilon=1e-5,
            dtype='float32'
        )

    def build(self, input_shape):
        self.C = int(input_shape[-1])

        self.q_dense, self.k_dense, self.v_dense = [
            layers.Dense(self.C, dtype='float32')
            for _ in range(3)
        ]

        self.spatial_bias = AdditiveSpatialBias(
            self.num_heads,
            self.window_size
        )

        self.mlp = MLPBlock(
            int(self.C * self.mlp_ratio),
            self.C
        )

    def call(self, x, guidance=None):
        orig_dtype = x.dtype
        x = tf.cast(x, tf.float32)

        shortcut = x

        x_norm = self.norm1(x)
        windows, meta = window_partition(
            x_norm,
            self.window_size
        )

        B_, N, C = (
            tf.shape(windows)[0],
            tf.shape(windows)[1],
            tf.shape(windows)[2]
        )

        guide_vec = None

        if guidance is not None:
            g_resized = tf.image.resize(
                tf.cast(guidance, tf.float32),
                (tf.shape(x_norm)[1], tf.shape(x_norm)[2])
            )

            guide_windows, _ = window_partition(
                g_resized,
                self.window_size
            )

            guide_vec = tf.reshape(
                guide_windows,
                [B_, N]
            )

        q = tf.transpose(
            tf.reshape(
                self.q_dense(windows),
                [B_, N, self.num_heads, C // self.num_heads]
            ),
            [0, 2, 1, 3]
        )

        k = tf.transpose(
            tf.reshape(
                self.k_dense(windows),
                [B_, N, self.num_heads, C // self.num_heads]
            ),
            [0, 2, 1, 3]
        )

        v = tf.transpose(
            tf.reshape(
                self.v_dense(windows),
                [B_, N, self.num_heads, C // self.num_heads]
            ),
            [0, 2, 1, 3]
        )

        q = tf.cast(q, tf.float32)
        k = tf.cast(k, tf.float32)
        v = tf.cast(v, tf.float32)

        attn_scores = tf.matmul(
            q,
            k,
            transpose_b=True
        ) / tf.sqrt(
            tf.cast(C // self.num_heads, tf.float32)
        )

        attn_scores = self.spatial_bias(
            attn_scores,
            guide_vec
        )

        attn_probs = tf.nn.softmax(
            attn_scores,
            axis=-1
        )

        attn_probs = tf.cast(
            attn_probs,
            tf.float32
        )

        x_attn = tf.matmul(
            attn_probs,
            v
        )

        x_attn = tf.reshape(
            tf.transpose(x_attn, [0, 2, 1, 3]),
            [B_, N, C]
        )

        x_merged = window_reverse(
            x_attn,
            self.window_size,
            meta
        )

        x = shortcut + x_merged
        x = self.norm2(x)
        x = x + self.mlp(x)

        return tf.cast(x, orig_dtype)


# ==========================================================
# Patch Operations
# ==========================================================

def patch_partition(x, patch_size=4, embed_dim=96):
    return layers.LayerNormalization(epsilon=1e-5)(
        layers.Conv2D(
            embed_dim,
            patch_size,
            patch_size,
            padding='same'
        )(x)
    )


def patch_merging(x, embed_dim):
    return layers.LayerNormalization(epsilon=1e-5)(
        layers.Conv2D(
            embed_dim * 2,
            2,
            2,
            padding='same'
        )(x)
    )


def patch_expanding(x, embed_dim):
    return layers.LayerNormalization(epsilon=1e-5)(
        layers.Conv2DTranspose(
            embed_dim // 2,
            2,
            2,
            padding='same'
        )(x)
    )


# ==========================================================
# Edge Integration
# ==========================================================

@keras.saving.register_keras_serializable()
class EdgeIntegration(layers.Layer):
    def __init__(
        self,
        percentile=92.0,
        gamma=0.6,
        blur_kernel=5,
        binary=True,
        threshold=0.2,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.percentile, self.gamma, self.binary, self.threshold = (
            percentile,
            gamma,
            binary,
            threshold
        )

        # Ensure the blur kernel size is odd.
        self.blur_kernel = int(
            blur_kernel
            if int(blur_kernel) % 2 == 1
            else int(blur_kernel) + 1
        )

    def call(self, x):
        x = tf.cast(x, tf.float32)

        # Select the first three channels or replicate a single channel.
        static_ch = x.shape[-1]

        if static_ch is not None:
            if static_ch >= 3:
                rgb = x[..., :3]
            else:
                rgb = tf.tile(
                    x[..., :1],
                    [1, 1, 1, 3]
                )
        else:
            channels = tf.shape(x)[-1]

            def _rgb():
                return x[..., :3]

            def _tile():
                return tf.tile(
                    x[..., :1],
                    [1, 1, 1, 3]
                )

            rgb = tf.cond(
                tf.greater_equal(channels, 3),
                _rgb,
                _tile
            )

        gray = tf.image.rgb_to_grayscale(rgb)

        # Sobel kernels.
        sobel_x = tf.constant(
            [
                [1, 0, -1],
                [2, 0, -2],
                [1, 0, -1]
            ],
            dtype=tf.float32
        )

        sobel_y = tf.constant(
            [
                [1, 2, 1],
                [0, 0, 0],
                [-1, -2, -1]
            ],
            dtype=tf.float32
        )

        sobel_x = tf.reshape(sobel_x, [3, 3, 1, 1])
        sobel_y = tf.reshape(sobel_y, [3, 3, 1, 1])

        gx = tf.nn.conv2d(
            gray,
            sobel_x,
            strides=[1, 1, 1, 1],
            padding="SAME"
        )

        gy = tf.nn.conv2d(
            gray,
            sobel_y,
            strides=[1, 1, 1, 1],
            padding="SAME"
        )

        grad = tf.sqrt(
            gx ** 2 + gy ** 2 + 1e-8
        )

        # Normalize per sample and apply gamma correction.
        max_per_sample = tf.reduce_max(
            grad,
            axis=[1, 2, 3],
            keepdims=True
        )

        grad_norm = grad / (max_per_sample + 1e-8)

        grad_gamma = tf.pow(
            grad_norm,
            tf.cast(self.gamma, grad_norm.dtype)
        )

        if self.binary:
            return tf.cast(
                tf.where(
                    grad_gamma > tf.cast(
                        self.threshold,
                        grad_gamma.dtype
                    ),
                    1.0,
                    0.0
                ),
                x.dtype
            )

        return tf.cast(
            grad_gamma,
            x.dtype
        )


# ==========================================================
# SwinUNet
# ==========================================================

def swin_unet(
    input_size=(512, 704, 4),
    num_classes=1,
    init_embed_dim=96,
    window_size=7,
    use_binary_edges=True,
    edge_threshold=0.2
):
    inputs = layers.Input(
        shape=input_size,
        name='model_input'
    )

    guidance = InputGuidance(
        name='input_guidance'
    )(inputs)

    x = patch_partition(
        inputs,
        4,
        init_embed_dim
    )

    skips, embed_dims, embed_dim = [], [], init_embed_dim

    edge_layers = [
        EdgeIntegration(
            binary=use_binary_edges,
            threshold=edge_threshold
        )
        for _ in range(3)
    ]

    edges = [
        edge(inputs)
        for edge in edge_layers
    ]

    # Encoder.
    for level in range(3):
        for _ in range(2):
            x = SwinTransformerBlock(
                num_heads=3 * (2 ** level),
                window_size=window_size
            )(x, guidance)

        skips.append(x)
        embed_dims.append(embed_dim)

        x = patch_merging(
            x,
            embed_dim
        )

        embed_dim *= 2

    # Bottleneck.
    for _ in range(2):
        x = SwinTransformerBlock(
            num_heads=24,
            window_size=window_size
        )(x, guidance)

    # Decoder.
    for level in reversed(range(3)):
        prev_dim = embed_dim
        embed_dim = embed_dims[level]

        x = patch_expanding(
            x,
            prev_dim
        )

        x = layers.Conv2D(
            embed_dim,
            1,
            padding='same'
        )(x)

        skip = skips[level]

        edge_resized = ResizeToMatch()([
            edges[level],
            skip
        ])

        skip = layers.Concatenate()([
            skip,
            edge_resized
        ])

        skip = layers.Conv2D(
            embed_dim,
            1,
            padding='same'
        )(skip)

        x = layers.Concatenate()([
            x,
            skip
        ])

        x = layers.Conv2D(
            embed_dim,
            1,
            padding='same'
        )(x)

        for _ in range(2):
            x = SwinTransformerBlock(
                num_heads=3 * (2 ** level),
                window_size=window_size
            )(x, guidance)

    # Final upsampling.
    x = patch_expanding(
        x,
        embed_dim * 2
    )

    x = patch_expanding(
        x,
        embed_dim
    )

    outputs = layers.Conv2D(
        num_classes,
        1,
        activation='sigmoid',
        dtype='float32'
    )(x)

    return keras.Model(
        inputs,
        outputs,
        name='TriNetra'
    )
