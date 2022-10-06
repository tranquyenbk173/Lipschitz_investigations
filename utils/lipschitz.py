from torch.autograd import grad as torch_grad
from torch.autograd import Variable
import torch


device = 'cuda' if torch.cuda.is_available() else 'cpu'

#----------- L_losss
def estimate_L_loss(model, loss_func, opt, data_batch, is_mse):
  # device = 'cuda' if torch.cuda.is_available() else 'cpu'
  max_norm = 0
  inputt, targett = data_batch
  inputt = Variable(inputt)
  inputt.requires_grad = True
  output = model(inputt)
  
  if is_mse:
        targets_onehot = target_2_onehot(targett).to(device)
        loss = loss_func(output, targets_onehot)
  else:
        loss = loss_func(output, targett)
            
  loss.backward()
  J = inputt.grad
  J = J.view(len(inputt), -1)
  norm_J = torch.norm(J, dim=1)
  max_norm_J = max(norm_J)
  if max_norm_J > max_norm:
    max_norm = max_norm_J

  inputt.grad.zero_()
  opt.zero_grad
  inputt.requires_grad = False

  return max_norm
  

#--------------L_model_derivative
def jacobian_matrix(y_flat, x, num_classes=10):
    y_flat = torch.transpose(y_flat, 0, 1)
    for i in range(num_classes):
        outputs_i = y_flat[i]
        if i==0:
            Jacobian = torch_grad(outputs=outputs_i, inputs=x,
                                grad_outputs=torch.ones(outputs_i.size()).cuda(),
                                create_graph=True, retain_graph=True)[0]
        else:
            temp = torch_grad(outputs=outputs_i, inputs=x,
                                grad_outputs=torch.ones(outputs_i.size()).cuda(),
                                create_graph=True, retain_graph=True)[0]
                                
            Jacobian = torch.cat([Jacobian, temp], axis=1)
    return Jacobian

def compute_grad_norm(model, inputs): #output at logit
  bs = inputs.shape[0]
  inputs = Variable(inputs, requires_grad=True)
  outputs = model(inputs)
  gradients = jacobian_matrix(outputs, inputs)
  gradients = gradients.view(bs, -1)
  gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
  return gradients_norm

def estimate_L_model(model, loss_func, data_batch):
#   model.eval()
  inputt, targett = data_batch
  inputt = Variable(inputt, requires_grad=True)
  output = model(inputt)
  bs=inputt.shape[0]

  norm_grad_fx = compute_grad_norm(model, inputt) #vecto, dim = (bs, 1)
  L_model = norm_grad_fx.max()
  
  return L_model
  
def target_2_onehot(targets):
        bs = targets.shape[0]
        onehot = torch.zeros(bs, 10)
        onehot = onehot + 1e-6
        for i, t in enumerate(targets):
            onehot[i][t] = 1
            
        return onehot

