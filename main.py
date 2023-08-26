import tensorflow as tf

class Attention2(tf.keras.layers.Layer):
    def __init__(self, heads, att_w, pos_w):
        super().__init__()
        self.H = heads
        self.a = att_w
        self.p = pos_w

    def build(self, input_shape):
        self.w = input_shape[-1]
        self.k = {}
        for name in ['k1', 'k2', 'k3']:
            self.k[name] = self.add_weight(
                shape=[self.H, self.w], trainable=True, initializer='random_uniform',
                name=name)
        self.pos = {}
        for name in ['a1', 'a2', 'b1', 'b2', 'c']:
            self.pos[name] = self.add_weight(
                shape = [self.H, self.p, 1], trainable=True, initializer='random_uniform',
                name=name)
        self.value_weight = self.add_weight(
            shape=[self.H, self.w, self.a], trainable=True, initializer='glorot_uniform',
            name='value_weight')
        self.output_weight = self.add_weight(
            shape=[self.H, self.w, self.a], trainable=True, initializer='glorot_uniform',
            name='output_weight')

    def call(self, x):
        n = tf.linspace(0.0, 1.0, tf.shape(x)[-2])
        p1 = tf.reduce_sum(tf.sin(
            self.pos['a1']*n + self.pos['b1']
            )*self.pos['c'], axis=-2)
        p2 = tf.reduce_sum(tf.sin(
            self.pos['a2']*n + self.pos['b2']
            )*self.pos['c'], axis=-2)

        cross = tf.einsum('bnw,hw->bhn', x, self.k['k1']) + p1
        diag = tf.einsum('bnw,hw->bhn', x, self.k['k2'])
        extra = tf.einsum('bnw,hw->bhn', x, self.k['k3'])
        values = tf.einsum('bnw,hwa->bhna', x, self.value_weight)

        cross = tf.exp(tf.clip_by_value(cross, -20.0, 20.0))[..., None]
        diag = tf.exp(tf.clip_by_value(diag, -20.0, 20.0))[..., None]
        p2 = tf.exp(tf.clip_by_value(p2, -20.0, 20.0))[..., None]
        extra = tf.exp(tf.clip_by_value(extra, -20.0, 20.0))[..., None]

        output = (
            tf.cumsum(cross*values, axis=2)*p2*extra + values*diag)/(
                tf.cumsum(cross, axis=2)*p2*extra + diag)

        output = tf.einsum('bhna,hwa->bnw', output, self.output_weight)
        return output
    
class FeedForward2(tf.keras.layers.Layer):
    def __init__(self, a, b):
        super().__init__()
        self.a = a
        self.b = b

    def build(self, input_shape):
        self.w = input_shape[-1]
        self.w1 = self.add_weight(
            shape = [self.w, self.a], trainable=True, initializer='glorot_uniform',
            name='w1')
        self.w2 = self.add_weight(
            shape = [self.w, self.b], trainable=True, initializer='glorot_uniform',
            name='w2')
        self.w3 = self.add_weight(
            shape = [self.a, self.b, self.w], trainable=True, initializer='glorot_uniform',
            name='w3')
        self.b1 = self.add_weight(
            shape = [self.a], trainable=True, initializer='zeros', name='b1')
        self.b2 = self.add_weight(
            shape = [self.b], trainable=True, initializer='zeros', name='b2')
        self.b3 = self.add_weight(
            shape = [self.w], trainable=True, initializer='zeros', name='b3')

    def call(self, x):
        a = tf.keras.activations.swish(tf.matmul(x, self.w1) + self.b1)
        b = tf.keras.activations.swish(tf.matmul(x, self.w2) + self.b2)
        return tf.einsum('bni,bnj,ijw->bnw', a, b, self.w3) + self.b3

class Model2(tf.keras.models.Model):
    def __init__(
        self, width, layers, heads, attention_width,
        position_width, linear_factor, dropout=0.0):
        super().__init__()
        self.width = width
        self.N = layers
        self.H = heads
        self.a = attention_width
        self.p = position_width
        self.l = linear_factor
        self.dropout = dropout

    def build(self, input_shape):
        assert input_shape[-1] == self.width
        self.atts = [Attention2(self.H, self.a, self.p) for _ in range(self.N)]
        self.lins = [FeedForward2(self.width, self.l) for _ in range(self.N)]
        self.norm = tf.keras.layers.LayerNormalization(axis=-1, scale=False, center=False)
        self.dropout_layer = tf.keras.layers.Dropout(self.dropout)

    def call(self, x, training=True):
        for i in range(self.N):
            x = x + self.dropout_layer(
                self.atts[i](self.norm(x)), training=training)
            x = x + self.dropout_layer(
                self.lins[i](self.norm(x)), training=training)
        return x
