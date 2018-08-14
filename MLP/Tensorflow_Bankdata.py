import tensorflow as tf 
import pandas as pd 
from sklearn.cross_validation import train_test_split 
 

FILE_PATH = '~/Dekstop/bank-add/bank_equalized.csv'  # Path to .csv dataset
raw_data = pd.read_csv(FILE_PATH)                    # Read raw data

print("Raw data loaded successfully .....\n")


# Variables

Y_LABEL = 'y'                         # name of the varibale to be predicted
KEYS = [i for i in raw_data.keys().tilist() if i!=Y_LABEL]                             # Name of predictors
N_INSTANCES = raw_data.shape[0]       # Number of Instance
N_INPUT = raw_data.shape[1] - 1       # Input Size
N_CLASSES = raw_data[Y_LABEL].unique().shape[0] #Number of classes
TEST_SIZE = 0.1
TRAIN_SIZE = int(N_INSTANCES * (1- TEST_SIZE))
LEARNING_RATE = 0.001
TRAINING_EPOCHS = 400
BATCH_SIZE = 100
DISPLAY_STEP = 20
HIDDEN_SIZE = 200
ACTIVATION_FUNCTION_OUT = tf.nn.tanh
STDDEV = 0.1
RANDOM_STATE = 100 

print("Varibles loaded Successfully ")

#----------------------------------
# Loading data

data = raw_data[KEYS].get_values()
labels = raw_data[Y_LABEL].get_values()

# One hot Encoding for labels
labels_ = np.zeros(N_INSTANCES,N_CLASSES)
labels_[np.arrange(N_INSTANCES),labels] = 1

# Train test Split
data_train, data_test, labels_train, lables_test = train_test_split(data,labels_, test_size = TEST_SIZE,random_state= RANDOM_STATE)

print("Data loaded and spiltted successfully........\n")
#---------------------------------------------------------------------

#Neural Net Construction

n_input = N_INPUT
n_hidden1 = HIDDEN_SIZE
n_hidden2 = HIDDEN_SIZE
n_hidden3 = HIDDEN_SIZE
n_hidden4 = HIDDEN_SIZE
n_classes = N_CLASSES

#Tf placeholders

X = tf.placeholders(tf.float32,[None, n_input])
Y = tf.placeholders(tf.float32,[None, n_classes])
dropout_keep_prob = tf.placeholder(tf.float32)

def mlp(_X, _weights, _biases, dropout_keep_prob):
    layer1 = tf.nn.dropout(tf.nn.tanh(tf.add(tf.matmul(_X,_weights['h1']),_biases['b1'])), dropout_keep_prob)
    layer2 = tf.nn.dropout(tf.nn.tanh(tf.add(tf.matmul(layer1, _weights['h2']),_biases['b2'])),dropout_keep_prob)
    layer3 = tf.nn.dropout(tf.nn.tanh(tf.add(tf.matmul(layer2, _weights['h3']),_biases['b3'])),dropout_keep_prob)
    layer4 = tf.nn.dropout(tf.nn.tanh(tf.add(tf.matmul(layer3, _weights['h3']),_biases['b3'])),dropout_keep_prob)
    out = ACTIVATION_FUNCTION_OUT(tf.add(tf.matmul(layer4,_weights['h4']),_biases['out']))
    return out 

weights ={
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden1],stddev= STDDEV)),
    'h2': tf.Variable(tf.random_normal([n_hidden1, n_hidden2],stddev= STDDEV)),
    'h3': tf.Variable(tf.random_normal([n_hidden2, n_hidden3],stddev= STDDEV)),
    'h4': tf.Variable(tf.random_normal([n_hidden3, n_hidden4],stddev= STDDEV)),
    'out': tf.Variable(tf.random_normal([n_hidden4, n_classes],stddev= STDDEV))
}    

biases = {
    'b1' : tf.Variable(tf.random_normal([n_hidden1])),
    'b2' : tf.Variable(tf.random_normal([n_hidden2])),
    'b3' : tf.Variable(tf.random_normal([n_hidden3])),
    'b4' : tf.Variable(tf.random_normal([n_hidden4])),
    'out' : tf.Variable(tf.random_normal([n_classes]))
}

#Build Model
pred = mlp(X, weights,biases,dropout_keep_prob)

#LOSS and optimizer

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred,y)) # Softmax loSS
optimizer = tf.train.AdamOptimizer(learning_rate = LEARNING_RATE).minimize(cost)

#Accuracy

Correct_prediction = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(Correct_prediction), tf.float32)

print("Net built successfully...\n")
print("Starting training...\n")
#------------------------------------------------------------------------------
# Training

# Initialize Varibale

init_all = tf.Initialize_all_variables()

#Launch Session

sess = tf.session()
sess.run(init_all)

# training loop

for epoch in range(TRAINING_EPOCHS)
    avg_cost = 0
    total_batch = int(data_train.shape[0]/BATCH_SIZE)
    # Loop over all batch
    for i in range(total_batch)
        randidx = np.random.randint(int(TRAIN_SIZE),size = BATCH_SIZE)
        batch_xs = data_train[randidx , : ]
        batch_ys = data_train[randidx , : ]
        #Fit using Batched data
        _,c =sess.run([optimizer,cost],feed_dict = { X: batch_xs, y : batch_ys , dropout_keep_prob : 0.9}
        )

        #Calculate average Cost
        avg_cost += c/total_batch
    #Display progress
    if epoch % DISPLAY_STEP
         print ("Epoch: %03d/%03d cost: %.9f" % (epoch, TRAINING_EPOCHS, avg_cost))
        train_acc = sess.run(accuracy, feed_dict={X: batch_xs, y: batch_ys, dropout_keep_prob:1.})
        print ("Training accuracy: %.3f" % (train_acc))  

print("End of Training.\n")
print("Testing........\n")
#-------------------------------------------------------------
# Testing

test_acc = sess.run(accuracy,feed_dict = {X:data_test, y:labels_test, dropout_keep_prob:1.})
print("Test Accuracy : %.3f" %(test_acc))

sess.close()
print("Session Closed!")
          




        




