import tensorflow as tf 


def load_model(model_path):
    interpreter = tf.lite.Interpreter(model_path)
    interpreter.allocate_tensors()
    return interpreter

'''
covnert input to numpmy array and preprocess with 
same format with pretrain model
'''
def preprocess(input):
    pass

def infer_model(model, data_test):
    input_test = preprocess(data_test)
    # get input object
    input_details = model.get_input_details()
    # get output object
    output_details = model.get_output_details()

    '''
    pass input_test to input object
    if you have many input just set is sequencely like
    model.set_tensor(input_details[0]['index'], input_test0)
    model.set_tensor(input_details[1]['index'], input_test1)
    '''

    model.set_tensor(input_details[0]['index'], input_test)
    # run model
    model.invoke()
    # get output
    output = model.get_tensor(output_details[0]['index'])
    return output

def postprocess(output):
    pass