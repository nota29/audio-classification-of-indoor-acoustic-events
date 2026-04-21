import tensorflow as tf


def training(model, opt, loss, callback_list, classweights, batchsize, train_data_size,
             val_data_size, epochs, train_data, val_data):
    model.compile(loss=loss, metrics=['accuracy'],
                  optimizer=opt)
    hist = model.fit(train_data, steps_per_epoch=-(-train_data_size // batchsize), epochs=epochs,
                     validation_data=val_data, batch_size=batchsize, validation_steps=-(-val_data_size // batchsize),
                     callbacks=callback_list, class_weight=classweights, verbose=1)
    return model, hist
