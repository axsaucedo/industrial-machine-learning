# ==============================================================================
# Deep Learning with Recurrent Neural Networks Workshop
# By Donald Whyte and Alejandro Saucedo
#
# Step 7:
# Adding Dropout to Reduce Overfitting
# ==============================================================================


    # save a checkpoint (every 500 batches)
    if step // 10 % _50_BATCHES == 0:
        saved_file = saver.save(sess, 'checkpoints/rnn_train_' + timestamp, global_step=step)
        print("Saved file: " + saved_file)
