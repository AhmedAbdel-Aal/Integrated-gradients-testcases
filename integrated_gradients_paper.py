import numpy as np
import torch

def integrated_gradients(
    inp, 
    target_label_index,
    predictions_and_gradients,
    baseline,
    model,
    steps=50):
  """Computes integrated gradients for a given network and prediction label.
  Integrated gradients is a technique for attributing a deep network's
  prediction to its input features. It was introduced by:
  https://arxiv.org/abs/1703.01365
  In addition to the integrated gradients tensor, the method also
  returns some additional debugging information for sanity checking
  the computation. See sanity_check_integrated_gradients for how this
  information is used.
  
  This method only applies to classification networks, i.e., networks 
  that predict a probability distribution across two or more class labels.
  
  Access to the specific network is provided to the method via a
  'predictions_and_gradients' function provided as argument to this method.
  The function takes a batch of inputs and a label, and returns the
  predicted probabilities of the label for the provided inputs, along with
  gradients of the prediction with respect to the input. Such a function
  should be easy to create in most deep learning frameworks.
  
  Args:
    inp: The specific input for which integrated gradients must be computed.
    target_label_index: Index of the target class for which integrated gradients
      must be computed.
    predictions_and_gradients: This is a function that provides access to the
      network's predictions and gradients. It takes the following
      arguments:
      - inputs: A batch of tensors of the same same shape as 'inp'. The first
          dimension is the batch dimension, and rest of the dimensions coincide
          with that of 'inp'.
      - target_label_index: The index of the target class for which gradients
        must be obtained.
      and returns:
      - predictions: Predicted probability distribution across all classes
          for each input. It has shape <batch, num_classes> where 'batch' is the
          number of inputs and num_classes is the number of classes for the model.
      - gradients: Gradients of the prediction for the target class (denoted by
          target_label_index) with respect to the inputs. It has the same shape
          as 'inputs'.
    baseline: [optional] The baseline input used in the integrated
      gradients computation. If None (default), the all zero tensor with
      the same shape as the input (i.e., 0*input) is used as the baseline.
      The provided baseline and input must have the same shape. 
    steps: [optional] Number of intepolation steps between the baseline
      and the input used in the integrated gradients computation. These
      steps along determine the integral approximation error. By default,
      steps is set to 50.
  Returns:
    integrated_gradients: The integrated_gradients of the prediction for the
      provided prediction label to the input. It has the same shape as that of
      the input.
      
    The following output is meant to provide debug information for sanity
    checking the integrated gradients computation.
    See also: sanity_check_integrated_gradients
    prediction_trend: The predicted probability distribution across all classes
      for the various (scaled) inputs considered in computing integrated gradients.
      It has shape <steps, num_classes> where 'steps' is the number of integrated
      gradient steps and 'num_classes' is the number of target classes for the
      model.
  """  
  if baseline is None:
    baseline = 0*inp
  assert(baseline.shape == inp.shape)

  # Scale input and compute gradients.
  scaled_inputs = [baseline + (float(i)/steps)*(inp-baseline) for i in range(0, steps+1)]
  grads, _ = predictions_and_gradients(model, scaled_inputs, target_label_index)  # shapes: <steps+1>, <steps+1, inp.shape>
  grads = torch.stack(grads)
  # Use trapezoidal rule to approximate the integral.
  # See Section 4 of the following paper for an accuracy comparison between
  # left, right, and trapezoidal IG approximations:
  # "Computing Linear Restrictions of Neural Networks", Matthew Sotoudeh, Aditya V. Thakur
  # https://arxiv.org/abs/1908.06214
  grads = (grads[:-1] + grads[1:]) / 2.0
  avg_grads = np.average(grads, axis=0)
  diff = (inp-baseline).detach()
  integrated_gradients = diff*avg_grads  # shape: <inp.shape>
  return integrated_gradients

def get_gradients_func(model, inputs, target_label_index):
    gradients = []
    predictions = []
    for input in inputs:
        prediction = model(input)
        target_output = prediction[0][target_label_index]
        predictions.append(target_output)
        gradients.append(torch.autograd.grad(target_output, input)[0])
    
    return gradients, predictions