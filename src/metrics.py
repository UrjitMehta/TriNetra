# ========== LOSS FUNCTIONS ==========

bce_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)


@register_keras_serializable()
def dice_coef(y_true, y_pred, smooth=1.0):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # Clip predictions for numerical stability.
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)

    # Flatten tensors.
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])

    # Compute intersection.
    intersection = tf.reduce_sum(y_true_f * y_pred_f)

    # Compute Dice score.
    dice = tf.math.divide_no_nan(
        2. * intersection + smooth,
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    )

    return dice


@register_keras_serializable()
def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


@register_keras_serializable(package="Custom")
def bce_dice_loss(y_true, y_pred):
    y_pred = tf.cast(y_pred, tf.float32)
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)

    return bce + dice


# ========== OTHER METRICS ==========

class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name="f1_score", threshold=0.1, **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.tp = self.add_weight(name="tp", initializer="zeros")
        self.fp = self.add_weight(name="fp", initializer="zeros")
        self.fn = self.add_weight(name="fn", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred > self.threshold, tf.float32)

        tp = tf.reduce_sum(y_true * y_pred)
        fp = tf.reduce_sum((1 - y_true) * y_pred)
        fn = tf.reduce_sum(y_true * (1 - y_pred))

        self.tp.assign_add(tp)
        self.fp.assign_add(fp)
        self.fn.assign_add(fn)

    def result(self):
        precision = self.tp / (self.tp + self.fp + K.epsilon())
        recall = self.tp / (self.tp + self.fn + K.epsilon())

        return 2 * precision * recall / (precision + recall + K.epsilon())

    def reset_states(self):
        for var in self.variables:
            K.set_value(var, 0)


class AFNR(tf.keras.metrics.Metric):
    def __init__(self, name="average_false_negative_ratio", threshold=0.5, **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.fn = self.add_weight(name="fn", initializer="zeros")
        self.tp = self.add_weight(name="tp", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred > self.threshold, tf.float32)

        fn = tf.reduce_sum(y_true * (1 - y_pred))
        tp = tf.reduce_sum(y_true * y_pred)

        self.fn.assign_add(fn)
        self.tp.assign_add(tp)

    def result(self):
        return self.fn / (self.fn + self.tp + K.epsilon())

    def reset_states(self):
        for var in self.variables:
            K.set_value(var, 0)


# Average Precision
ap_metric = tf.keras.metrics.AUC(curve="PR", name="average_precision")
