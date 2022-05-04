import tensorflow as tf

beta = tf.random.normal([200, 10])
inputs = tf.keras.layers.Input(200)
beta=tf.square(beta, name=None)

beta_sum=tf.reduce_sum(beta, -1, keepdims=True)
u=tf.divide(beta, beta_sum, name=None)
inputs_new=tf.expand_dims(inputs, -1)

for i in range(200):
    if i==0:
        mass_prototype_i=tf.multiply(u[i,:], inputs_new[:,i], name=None)
        #print("标记")
        #print(mass_prototype_i)
        mass_prototype=tf.expand_dims(mass_prototype_i, -2)
    if i>0:
        mass_prototype_i=tf.expand_dims(tf.multiply(u[i,:], inputs_new[:,i], name=None), -2)
        mass_prototype=tf.concat([mass_prototype, mass_prototype_i], -2)
mass_prototype=tf.convert_to_tensor(mass_prototype)


mass_omega_sum = tf.reduce_sum(mass_prototype, -1, keepdims=True) # (None, 200, 1)
mass_omega_sum = tf.subtract(1. , mass_omega_sum[:,:,0], name=None)  # (
mass_omega_sum = tf.expand_dims(mass_omega_sum, -1)
mass_with_omega = tf.concat([mass_prototype, mass_omega_sum], -1)

inputs = mass_with_omega

m1 = inputs[:,0,:]  # (None, 11)
omega1 = tf.expand_dims(inputs[:,0,-1], -1)  # (None, 1 )
for i in range(199):
    m2 = inputs[:,(i+1),:]
    omega2 = tf.expand_dims(inputs[:,(i+1),-1], -1) # (None, 1)
    combine1 = tf.multiply(m1, m2, name=None) # (None, 11)
    combine2 = tf.multiply(m1, omega2, name=None)  # (Nobe, 11)
    combine3 = tf.multiply(omega1, m2, name=None)
    combine1_2 = tf.add(combine1, combine2, name=None)
    combine2_3 = tf.add(combine1_2, combine3,name=None)
    combine2_3 = combine2_3 / tf.reduce_sum(combine2_3, axis=-1, keepdims=True)
    m1 = combine2_3
    omega1 = tf.expand_dims(combine2_3[:,-1], -1)



