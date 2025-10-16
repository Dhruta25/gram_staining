import tensorflow as tf

# small helpers (student style)

def conv2d_block(filters, kernel_size=3, strides=1, name=None):
    """Return a small Conv2D -> Bias add wrapper (student-friendly)."""
    return tf.keras.layers.Conv2D(filters=filters,
                                  kernel_size=kernel_size,
                                  strides=strides,
                                  padding='same',
                                  kernel_initializer=tf.keras.initializers.VarianceScaling(),
                                  bias_initializer='zeros',
                                  name=name)

def leaky_relu(x, alpha=0.1):
    return tf.nn.leaky_relu(x, alpha=alpha)

def silu(x):
    return x * tf.sigmoid(x)


# Channel Attention Block

class ChannelAttention(tf.keras.layers.Layer):
    def __init__(self, channels, ratio=8, name='channel_attention'):
        super().__init__(name=name)
        self.channels = channels
        self.ratio = ratio
        self.pool = tf.keras.layers.GlobalAveragePooling2D(keepdims=True)
        # small MLP with bottleneck
        self.fc1 = tf.keras.layers.Conv2D(filters=channels // ratio, kernel_size=1, padding='same',
                                          activation='relu', name=name + '_fc1')
        self.fc2 = tf.keras.layers.Conv2D(filters=channels, kernel_size=1, padding='same',
                                          activation='sigmoid', name=name + '_fc2')

    def call(self, x):
        # x shape: (B, H, W, C)
        skip = x
        x = self.pool(x)                     # (B,1,1,C)
        x = self.fc1(x)                      # (B,1,1,C/ratio)
        x = self.fc2(x)                      # (B,1,1,C) -> sigmoid weights
        return skip * x                      # channel-wise scaling

# -------------------------
# Residual Channel Attention Block (RCA)
# -------------------------
class RCABlock(tf.keras.layers.Layer):
    def __init__(self, channels, kernel_size=3, ratio=8, name='rca'):
        super().__init__(name=name)
        self.conv1 = conv2d_block(channels, kernel_size, strides=1, name=name + '_c1')
        self.conv2 = conv2d_block(channels, kernel_size, strides=1, name=name + '_c2')
        self.ca = ChannelAttention(channels, ratio=ratio, name=name + '_ca')

    def call(self, x):
        skip = x
        x = self.conv1(x)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.ca(x)
        return x + skip

# -------------------------
# Simple residual block used in discriminator
# -------------------------
class SimpleResBlock(tf.keras.layers.Layer):
    def __init__(self, channels, name='resblock'):
        super().__init__(name=name)
        self.conv1 = conv2d_block(channels, 3, 1, name=name + '_c1')
        self.conv2 = conv2d_block(channels, 3, 1, name=name + '_c2')

    def call(self, x):
        skip = x
        x = self.conv1(x)
        x = leaky_relu(x)
        x = self.conv2(x)
        x = leaky_relu(x)
        return x + skip

# -------------------------
# Down block used by generator (encoder)
# -------------------------
class DownBlock(tf.keras.layers.Layer):
    def __init__(self, in_ch, out_ch, name='down'):
        super().__init__(name=name)
        self.conv1 = conv2d_block(in_ch, 3, 1, name=name + '_c1')
        self.conv2 = conv2d_block(out_ch, 3, 1, name=name + '_c2')

    def call(self, x):
        # note: we keep a skip connection padded if needed like original
        x1 = self.conv1(x)
        x1 = leaky_relu(x1)
        x2 = self.conv2(x1)
        x2 = leaky_relu(x2)
        # pad the input channels to match out_ch if needed
        in_ch = tf.shape(x)[-1]
        out_ch = tf.shape(x2)[-1]
        if in_ch != out_ch:
            # pad on channel axis
            pad_ch = out_ch - in_ch
            x_padded = tf.pad(x, [[0,0],[0,0],[0,0],[0,pad_ch]])
        else:
            x_padded = x
        # residual-like combine (similar to your original conv2 + tmp)
        combined = x2 + x_padded
        # downsample by average pooling
        pooled = tf.nn.avg_pool2d(combined, ksize=2, strides=2, padding='SAME')
        return pooled, combined  # return pooled output and the skip feature

# -------------------------
# Up block used by generator (decoder)
# -------------------------
class UpBlock(tf.keras.layers.Layer):
    def __init__(self, in_ch, out_ch, name='up'):
        super().__init__(name=name)
        self.conv1 = conv2d_block(in_ch*2, in_ch, 3, name=name + '_c1')  # after concat channels ~ 2*in_ch
        self.conv2 = conv2d_block(in_ch, out_ch, 3, name=name + '_c2')

    def call(self, x, skip_feature, out_size=None):
        # simple bilinear upsample then concat with skip
        if out_size is None:
            # infer size from skip_feature
            out_h = tf.shape(skip_feature)[1]
            out_w = tf.shape(skip_feature)[2]
        else:
            out_h, out_w = out_size, out_size
        x_up = tf.image.resize(x, [out_h, out_w], method='bilinear')
        x_cat = tf.concat([x_up, skip_feature], axis=-1)
        x = self.conv1(x_cat)
        x = leaky_relu(x)
        x = self.conv2(x)
        x = leaky_relu(x)
        return x

# -------------------------
# Generator (simple encoder-decoder with skips)
# -------------------------
class Generator(tf.keras.Model):
    def __init__(self, n_levels=3, init_channels=32, out_channels=3, image_size=256, name='Generator'):
        super().__init__(name=name)
        self.n_levels = n_levels
        self.init_channels = init_channels
        self.image_size = image_size

        # initial conv
        self.start_conv = conv2d_block(init_channels, 3, 1, name='g_start')

        # build down blocks
        self.down_blocks = []
        self.up_blocks = []
        ch = init_channels
        for i in range(n_levels):
            self.down_blocks.append(DownBlock(ch, ch*2, name=f'down{i+1}'))
            ch *= 2

        # center conv
        self.center_conv = conv2d_block(ch, 3, 1, name='g_center')

        # build up blocks (reverse)
        for i in range(n_levels):
            self.up_blocks.append(UpBlock(ch//2, ch//4 if i < n_levels-1 else ch//2, name=f'up{i+1}'))
            ch = ch // 2

        # last conv to image
        self.last_conv = conv2d_block(out_channels, 3, 1, name='g_last')

    def call(self, x):
        skips = []
        x = self.start_conv(x)
        x = leaky_relu(x)

        # encoder
        for down in self.down_blocks:
            x, skip = down(x)
            skips.append(skip)

        # center
        x = self.center_conv(x)
        x = leaky_relu(x)

        # decoder (use skips in reverse order)
        for up, skip in zip(self.up_blocks, reversed(skips)):
            # compute target size from skip
            size_h = tf.shape(skip)[1]
            x = up(x, skip_feature=skip, out_size=size_h)

        # output
        x = self.last_conv(x)
        return x

# -------------------------
# Discriminator (simple conv + res blocks + fc)
# -------------------------
class Discriminator(tf.keras.Model):
    def __init__(self, base_channels=32, name='Discriminator'):
        super().__init__(name=name)
        self.start_conv = conv2d_block(base_channels, 3, 1, name='d_start')
        # a few normal blocks with downsampling to reduce spatial resolution
        self.norm_blocks = []
        ch = base_channels
        for i in range(5):
            self.norm_blocks.append(conv2d_block(ch*2, 3, strides=2, name=f'd_nblock{i}'))
            ch *= 2
            # a small residual block for local refinement
            self.norm_blocks.append(SimpleResBlock(ch, name=f'd_res{i}'))

        self.global_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.fc1 = tf.keras.layers.Dense(ch, activation=None, name='d_fc1',
                                         kernel_initializer=tf.keras.initializers.VarianceScaling())
        self.fc2 = tf.keras.layers.Dense(1, activation='sigmoid', name='d_fc2',
                                         kernel_initializer=tf.keras.initializers.VarianceScaling())

    def call(self, x):
        x = self.start_conv(x)
        x = leaky_relu(x)
        for blk in self.norm_blocks:
            x = blk(x)
        x = self.global_pool(x)
        x = self.fc1(x)
        x = leaky_relu(x)
        x = self.fc2(x)
        return x

# -------------------------
# Example usage (student friendly)
# -------------------------
if __name__ == '__main__':
    # quick smoke test to check shapes
    batch = 2
    img_size = 128
    in_ch = 5
    x = tf.random.normal([batch, img_size, img_size, in_ch])

    G = Generator(n_levels=3, init_channels=16, out_channels=3, image_size=img_size)
    out = G(x)
    print('Generator output shape:', out.shape)   # expected (B, H, W, 3)

    D = Discriminator(base_channels=16)
    d_out = D(out)
    print('Discriminator output shape:', d_out.shape)  # expected (B,1)