# One hiddent layer neural network with optional dropout and l2 regularization

# Helper functions

# def init_weights(shape, name):
#   return tf.Variable(tf.random_normal(shape, stddev=0.01), name=name)

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
                    / predictions.shape[0])

# This network is the same as the previous one except with an extra hidden layer + dropout
def model(X, l1, b1, l2, b2, p_dropout):
    # Add layer name scopes for better graph visualization
    with tf.name_scope("input_layer"):
        h = tf.nn.relu(tf.matmul(X, l1) + b1)
    with tf.name_scope("hidden layer"):
        h = tf.nn.dropout(h, p_dropout)
        h2 = tf.matmul(h, l2) + b_2
        return(h2)

# Finction for training model
def train(
    tf_train_dataset,
    tf_valid_dataset,
    tf_test_dataset,
    logdir = '"./logs/nn_logs',
    hidden_nodes = 1028,
    batch_size = 128,
    num_steps = 3001,
    num_labels = 10,
    image_size = 784,
    dropout = 1.0,
    l2 = 0.0):

    # Input data
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size * image_size))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)
    l2 = tf.placeholder(tf.float32)
    p_dropout = tf.placeholder(tf.float32)
    
    # Variables

    #Input layer
    weights_1 = tf.Variable(
        tf.truncated_normal([image_size * image_size, hidden_nodes]))
    biases_1 = tf.Variable(tf.zeros([hidden_nodes]))
    
    #Hidden layer
    weights_2 = tf.Variable(
        tf.truncated_normal([hidden_nodes, num_labels]))
    biases_2 = tf.Variable(tf.zeros([num_labels]))
    
    #Visualization
    tf.histogram_summary("input_layer_weights", weights_1)
    tf.histogram_summary("hidden_layer_weights", weights_2)
    tf.histogram_summary("input_layer_biases", biases_1)
    tf.histogram_summary("hidden_layer_biases", biases_2)    

    #Create Model
    logits = model(
        tf_train_dataset,
        weights_1,
        biases_1,
        weights_2,
        biases_2,
        p_dropout)

    #Create loss function
    with tf.name_scope("loss"):
        loss = tf.reduce_mean(
            softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits) + 
            l2 * tf.nn.l2_loss(weights_1) + l2 * tf.nn.l2_loss(weights_2)
        optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
        # Add scalar summary for cost tensor
        tf.scalar_summary("loss", loss)

    #Measure accuracy
    with tf.name_scope("accuracy"):
        
        val_acc = accuracy(model, tf_valid_labels)
        test_acc = accuracy(model, tf_test_labels)

        # Add scalar summary for accuracy tensor
        tf.scalar_summary("validation_accuracy", val_acc)
        tf.scalar_summary("test_accuracy", test_acc)

    #Create a session
    with tf.Session() as sess:
        
        #Сreate a log writer. run 'tensorboard --logdir=./logs/nn_logs'
        writer = tf.train.SummaryWriter("./logs/nn_logs", sess.graph) # for 0.8
        merged = tf.merge_all_summaries()

        #Iinitialize all variables
        tf.initialize_all_variables().run()

        #Train the  model
        for step in range(num_steps):

            # Pick an offset within the training data, which has been randomized.
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)

            #Generate a minibatch.
            batch_data = train_dataset[offset:(offset + batch_size), :]
            batch_labels = train_labels[offset:(offset + batch_size), :]

            #Prepare a dictionary telling the session where to feed the minibatch.
            feed_dict = {
                tf_train_dataset : batch_data,
                tf_train_labels : batch_labels, 
                l2: l2_loss
            }
            

            _, l, acc = session.run([optimizer, loss, val_acc], feed_dict=feed_dict)


            for start, end in zip(range(0, len(trX), 128), range(128, len(trX)+1, 128)):
                sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end],
                                              p_keep_input: 0.8, p_keep_hidden: 0.5})
            summary, acc = sess.run([merged, acc_op], feed_dict={X: teX, Y: teY,
                                              p_keep_input: 1.0, p_keep_hidden: 1.0})
            writer.add_summary(summary, i)  # Write summary
            print(i, acc)                   # Report the accuracy




    with tf.Session(graph=graph_2l) as session:
        tf.initialize_all_variables().run()
        print("Initialized")
        for step in range(num_steps):

            # Pick an offset within the training data, which has been randomized.
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)

            # Generate a minibatch.
            batch_data = train_dataset[offset:(offset + batch_size), :]
            batch_labels = train_labels[offset:(offset + batch_size), :]

            # Prepare a dictionary telling the session where to feed the minibatch.
            feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels, tf_l2_loss: l2_loss}
            _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)

            if (step % period_stat == 0):
                history["steps"].append(step)
                history["validation"].append(accuracy(valid_prediction.eval(), valid_labels))
                history["minibatch"].append(accuracy(predictions, batch_labels))
                history["test"].append(accuracy(test_prediction.eval(), test_labels))

        plot(history)
        print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))

try_loss = [0.0001, 0.001, 0.01, 0.03, 0.1, 0.3]
for l2_loss in try_loss:
    train_neural_n(l2_loss)
