import os
import sys

import torch
from torch.autograd import Variable
import torch.nn as nn

from pyprojroot import here as project_root


sys.path.insert(0, str(project_root()))
from qpth_local.qpth.qp import QPFunction
# from qpth.qp import QPFunction


def computeGramMatrix(A, B):
  """
  Constructs a linear kernel matrix between A and B.
  We assume that each row in A and B represents a d-dimensional feature vector.

  Parameters:
    A:  a (n_batch, n, d) Tensor.
    B:  a (n_batch, m, d) Tensor.
  Returns: a (n_batch, n, m) Tensor.
  """

  assert (A.dim() == 3)
  assert (B.dim() == 3)
  assert (A.size(0) == B.size(0) and A.size(2) == B.size(2))

  return torch.bmm(A, B.transpose(1, 2))


def binv(b_mat):
  """
  Computes an inverse of each matrix in the batch.
  Pytorch 0.4.1 does not support batched matrix inverse.
  Hence, we are solving AX=I.

  Parameters:
    b_mat:  a (n_batch, n, n) Tensor.
  Returns: a (n_batch, n, n) Tensor.
  """

  id_matrix = b_mat.new_ones(b_mat.size(-1)).diag().expand_as(b_mat).cuda()
  b_inv, _ = torch.gesv(id_matrix, b_mat)

  return b_inv


def one_hot(indices, depth):
  """
  Returns a one-hot tensor.
  This is a PyTorch equivalent of Tensorflow's tf.one_hot.

  Parameters:
    indices:  a (n_batch, m) Tensor or (m) Tensor.
    depth: a scalar. Represents the depth of the one hot dimension.
  Returns: a (n_batch, m, depth) Tensor or (m, depth) Tensor.
  """

  encoded_indicies = torch.zeros(indices.size() + torch.Size([depth])).cuda()
  index = indices.view(indices.size() + torch.Size([1]))
  encoded_indicies = encoded_indicies.scatter_(1, index, 1)

  return encoded_indicies


def batched_kronecker(matrix1, matrix2):
  matrix1_flatten = matrix1.reshape(matrix1.size()[0], -1)
  matrix2_flatten = matrix2.reshape(matrix2.size()[0], -1)
  return torch.bmm(matrix1_flatten.unsqueeze(2), matrix2_flatten.unsqueeze(1)).reshape(
    [matrix1.size()[0]] + list(matrix1.size()[1:]) + list(matrix2.size()[1:])).permute([0, 1, 3, 2, 4]).reshape(
    matrix1.size(0), matrix1.size(1) * matrix2.size(1), matrix1.size(2) * matrix2.size(2))


def MetaOptNetHead_Ridge(query, support, support_labels, n_way, n_shot, lambda_reg=50.0, double_precision=False):
  """
  Fits the support set with ridge regression and
  returns the classification score on the query set.

  Parameters:
    query:  a (tasks_per_batch, n_query, d) Tensor.
    support:  a (tasks_per_batch, n_support, d) Tensor.
    support_labels: a (tasks_per_batch, n_support) Tensor.
    n_way: a scalar. Represents the number of classes in a few-shot classification task.
    n_shot: a scalar. Represents the number of support examples given per class.
    lambda_reg: a scalar. Represents the strength of L2 regularization.
  Returns: a (tasks_per_batch, n_query, n_way) Tensor.
  """

  tasks_per_batch = query.size(0)
  n_support = support.size(1)
  n_query = query.size(1)

  assert (query.dim() == 3)
  assert (support.dim() == 3)
  assert (query.size(0) == support.size(0) and query.size(2) == support.size(2))
  assert (n_support == n_way * n_shot)  # n_support must equal to n_way * n_shot

  # Here we solve the dual problem:
  # Note that the classes are indexed by m & samples are indexed by i.
  # min_{\alpha}  0.5 \sum_m ||w_m(\alpha)||^2 + \sum_i \sum_m e^m_i alpha^m_i

  # where w_m(\alpha) = \sum_i \alpha^m_i x_i,

  # \alpha is an (n_support, n_way) matrix
  kernel_matrix = computeGramMatrix(support, support)
  kernel_matrix += lambda_reg * torch.eye(n_support).expand(tasks_per_batch, n_support, n_support).cuda()

  block_kernel_matrix = kernel_matrix.repeat(n_way, 1, 1)  # (n_way * tasks_per_batch, n_support, n_support)

  support_labels_one_hot = one_hot(support_labels.view(tasks_per_batch * n_support),
                                   n_way)  # (tasks_per_batch * n_support, n_way)
  support_labels_one_hot = support_labels_one_hot.transpose(0, 1)  # (n_way, tasks_per_batch * n_support)
  support_labels_one_hot = support_labels_one_hot.reshape(n_way * tasks_per_batch,
                                                          n_support)  # (n_way*tasks_per_batch, n_support)

  G = block_kernel_matrix
  e = -2.0 * support_labels_one_hot

  # This is a fake inequlity constraint as qpth does not support QP without an inequality constraint.
  id_matrix_1 = torch.zeros(tasks_per_batch * n_way, n_support, n_support)
  C = Variable(id_matrix_1)
  h = Variable(torch.zeros((tasks_per_batch * n_way, n_support)))
  dummy = Variable(torch.Tensor()).cuda()  # We want to ignore the equality constraint.

  if double_precision:
    G, e, C, h = [x.double().cuda() for x in [G, e, C, h]]

  else:
    G, e, C, h = [x.float().cuda() for x in [G, e, C, h]]

  # Solve the following QP to fit SVM:
  #        \hat z =   argmin_z 1/2 z^T G z + e^T z
  #                 subject to Cz <= h
  # We use detach() to prevent backpropagation to fixed variables.
  qp_sol = QPFunction(verbose=False)(G, e.detach(), C.detach(), h.detach(), dummy.detach(), dummy.detach())
  # qp_sol = QPFunction(verbose=False)(G, e.detach(), dummy.detach(), dummy.detach(), dummy.detach(), dummy.detach())

  # qp_sol (n_way*tasks_per_batch, n_support)
  qp_sol = qp_sol.reshape(n_way, tasks_per_batch, n_support)
  # qp_sol (n_way, tasks_per_batch, n_support)
  qp_sol = qp_sol.permute(1, 2, 0)
  # qp_sol (tasks_per_batch, n_support, n_way)

  # Compute the classification score.
  compatibility = computeGramMatrix(support, query)
  compatibility = compatibility.float()
  compatibility = compatibility.unsqueeze(3).expand(tasks_per_batch, n_support, n_query, n_way)
  qp_sol = qp_sol.reshape(tasks_per_batch, n_support, n_way)
  logits = qp_sol.float().unsqueeze(2).expand(tasks_per_batch, n_support, n_query, n_way)
  logits = logits * compatibility
  logits = torch.sum(logits, 1)

  return logits


def R2D2Head(query, support, support_labels, n_way, n_shot, l2_regularizer_lambda=50.0):
  """
  Fits the support set with ridge regression and
  returns the classification score on the query set.

  This model is the classification head described in:
  Meta-learning with differentiable closed-form solvers
  (Bertinetto et al., in submission to NIPS 2018).

  Parameters:
    query:  a (tasks_per_batch, n_query, d) Tensor.
    support:  a (tasks_per_batch, n_support, d) Tensor.
    support_labels: a (tasks_per_batch, n_support) Tensor.
    n_way: a scalar. Represents the number of classes in a few-shot classification task.
    n_shot: a scalar. Represents the number of support examples given per class.
    l2_regularizer_lambda: a scalar. Represents the strength of L2 regularization.
  Returns: a (tasks_per_batch, n_query, n_way) Tensor.
  """

  tasks_per_batch = query.size(0)
  n_support = support.size(1)

  assert (query.dim() == 3)
  assert (support.dim() == 3)
  assert (query.size(0) == support.size(0) and query.size(2) == support.size(2))
  assert (n_support == n_way * n_shot)  # n_support must equal to n_way * n_shot

  support_labels_one_hot = one_hot(support_labels.view(tasks_per_batch * n_support), n_way)
  support_labels_one_hot = support_labels_one_hot.view(tasks_per_batch, n_support, n_way)

  id_matrix = torch.eye(n_support).expand(tasks_per_batch, n_support, n_support).cuda()

  # Compute the dual form solution of the ridge regression.
  # W = X^T(X X^T - lambda * I)^(-1) Y
  ridge_sol = computeGramMatrix(support, support) + l2_regularizer_lambda * id_matrix
  ridge_sol = binv(ridge_sol)
  ridge_sol = torch.bmm(support.transpose(1, 2), ridge_sol)
  ridge_sol = torch.bmm(ridge_sol, support_labels_one_hot)

  # Compute the classification score.
  # score = W X
  logits = torch.bmm(query, ridge_sol)

  return logits


def MetaOptNetHead_SVM_He(query, support, support_labels, n_way, n_shot, C_reg=0.01, double_precision=False):
  """
  Fits the support set with multi-class SVM and
  returns the classification score on the query set.

  This is the multi-class SVM presented in:
  A simplified multi-class support vector machine with reduced dual optimization
  (He et al., Pattern Recognition Letter 2012).

  This SVM is desirable because the dual variable of size is n_support
  (as opposed to n_way*n_support in the Weston&Watkins or Crammer&Singer multi-class SVM).
  This model is the classification head that we have initially used for our project.
  This was dropped since it turned out that it performs suboptimally on the meta-learning scenarios.

  Parameters:
    query:  a (tasks_per_batch, n_query, d) Tensor.
    support:  a (tasks_per_batch, n_support, d) Tensor.
    support_labels: a (tasks_per_batch, n_support) Tensor.
    n_way: a scalar. Represents the number of classes in a few-shot classification task.
    n_shot: a scalar. Represents the number of support examples given per class.
    C_reg: a scalar. Represents the cost parameter C in SVM.
  Returns: a (tasks_per_batch, n_query, n_way) Tensor.
  """

  tasks_per_batch = query.size(0)
  n_support = support.size(1)
  n_query = query.size(1)

  assert (query.dim() == 3)
  assert (support.dim() == 3)
  assert (query.size(0) == support.size(0) and query.size(2) == support.size(2))
  assert (n_support == n_way * n_shot)  # n_support must equal to n_way * n_shot

  kernel_matrix = computeGramMatrix(support, support)

  V = (support_labels * n_way - torch.ones(tasks_per_batch, n_support, n_way).cuda()) / (n_way - 1)
  G = computeGramMatrix(V, V).detach()
  G = kernel_matrix * G

  e = Variable(-1.0 * torch.ones(tasks_per_batch, n_support))
  id_matrix = torch.eye(n_support).expand(tasks_per_batch, n_support, n_support)
  C = Variable(torch.cat((id_matrix, -id_matrix), 1))
  h = Variable(torch.cat((C_reg * torch.ones(tasks_per_batch, n_support), torch.zeros(tasks_per_batch, n_support)), 1))
  dummy = Variable(torch.Tensor()).cuda()  # We want to ignore the equality constraint.

  if double_precision:
    G, e, C, h = [x.double().cuda() for x in [G, e, C, h]]
  else:
    G, e, C, h = [x.cuda() for x in [G, e, C, h]]

  # Solve the following QP to fit SVM:
  #        \hat z =   argmin_z 1/2 z^T G z + e^T z
  #                 subject to Cz <= h
  # We use detach() to prevent backpropagation to fixed variables.
  qp_sol = QPFunction(verbose=False)(G, e.detach(), C.detach(), h.detach(), dummy.detach(), dummy.detach())

  # Compute the classification score.
  compatibility = computeGramMatrix(query, support)
  compatibility = compatibility.float()

  logits = qp_sol.float().unsqueeze(1).expand(tasks_per_batch, n_query, n_support)
  logits = logits * compatibility
  logits = logits.view(tasks_per_batch, n_query, n_shot, n_way)
  logits = torch.sum(logits, 2)

  return logits


def ProtoNetHead(query, support, support_labels, n_way, n_shot, normalize=True):
  """
  Constructs the prototype representation of each class(=mean of support vectors of each class) and
  returns the classification score (=L2 distance to each class prototype) on the query set.

  This model is the classification head described in:
  Prototypical Networks for Few-shot Learning
  (Snell et al., NIPS 2017).

  Parameters:
    query:  a (tasks_per_batch, n_query, d) Tensor.
    support:  a (tasks_per_batch, n_support, d) Tensor.
    support_labels: a (tasks_per_batch, n_support) Tensor.
    n_way: a scalar. Represents the number of classes in a few-shot classification task.
    n_shot: a scalar. Represents the number of support examples given per class.
    normalize: a boolean. Represents whether if we want to normalize the distances by the embedding dimension.
  Returns: a (tasks_per_batch, n_query, n_way) Tensor.
  """

  tasks_per_batch = query.size(0)
  n_support = support.size(1)
  n_query = query.size(1)
  d = query.size(2)

  assert (query.dim() == 3)
  assert (support.dim() == 3)
  assert (query.size(0) == support.size(0) and query.size(2) == support.size(2))
  assert (n_support == n_way * n_shot)  # n_support must equal to n_way * n_shot

  support_labels_one_hot = one_hot(support_labels.view(tasks_per_batch * n_support), n_way)
  support_labels_one_hot = support_labels_one_hot.view(tasks_per_batch, n_support, n_way)

  # From:
  # https://github.com/gidariss/FewShotWithoutForgetting/blob/master/architectures/PrototypicalNetworksHead.py
  # ************************* Compute Prototypes **************************
  labels_train_transposed = support_labels_one_hot.transpose(1, 2)
  # Batch matrix multiplication:
  #   prototypes = labels_train_transposed * features_train ==>
  #   [batch_size x nKnovel x num_channels] =
  #       [batch_size x nKnovel x num_train_examples] * [batch_size * num_train_examples * num_channels]
  prototypes = torch.bmm(labels_train_transposed, support)
  # Divide with the number of examples per novel category.
  prototypes = prototypes.div(
    labels_train_transposed.sum(dim=2, keepdim=True).expand_as(prototypes)
  )

  # Distance Matrix Vectorization Trick
  AB = computeGramMatrix(query, prototypes)
  AA = (query * query).sum(dim=2, keepdim=True)
  BB = (prototypes * prototypes).sum(dim=2, keepdim=True).reshape(tasks_per_batch, 1, n_way)
  logits = AA.expand_as(AB) - 2 * AB + BB.expand_as(AB)
  logits = -logits

  if normalize:
    logits = logits / d

  return logits


def MetaOptNetHead_SVM_CS(query, support, support_labels, n_way, n_shot, C_reg=0.1, double_precision=False, maxIter=15):
  """
  Fits the support set with multi-class SVM and
  returns the classification score on the query set.

  This is the multi-class SVM presented in:
  On the Algorithmic Implementation of Multiclass Kernel-based Vector Machines
  (Crammer and Singer, Journal of Machine Learning Research 2001).

  This model is the classification head that we use for the final version.
  Parameters:
    query:  a (tasks_per_batch, n_query, d) Tensor.
    support:  a (tasks_per_batch, n_support, d) Tensor.
    support_labels: a (tasks_per_batch, n_support) Tensor.
    n_way: a scalar. Represents the number of classes in a few-shot classification task.
    n_shot: a scalar. Represents the number of support examples given per class.
    C_reg: a scalar. Represents the cost parameter C in SVM.
  Returns: a (tasks_per_batch, n_query, n_way) Tensor.
  """

  tasks_per_batch = query.size(0)
  n_support = support.size(1)
  n_query = query.size(1)

  assert (query.dim() == 3)
  assert (support.dim() == 3)
  assert (query.size(0) == support.size(0) and query.size(2) == support.size(2))
  assert (n_support == n_way * n_shot)  # n_support must equal to n_way * n_shot

  # Here we solve the dual problem:
  # Note that the classes are indexed by m & samples are indexed by i.
  # min_{\alpha}  0.5 \sum_m ||w_m(\alpha)||^2 + \sum_i \sum_m e^m_i alpha^m_i
  # s.t.  \alpha^m_i <= C^m_i \forall m,i , \sum_m \alpha^m_i=0 \forall i

  # where w_m(\alpha) = \sum_i \alpha^m_i x_i,
  # and C^m_i = C if m  = y_i,
  # C^m_i = 0 if m != y_i.
  # This borrows the notation of liblinear.

  # \alpha is an (n_support, n_way) matrix
  kernel_matrix = computeGramMatrix(support, support)

  id_matrix_0 = torch.eye(n_way).expand(tasks_per_batch, n_way, n_way).cuda()
  block_kernel_matrix = batched_kronecker(kernel_matrix, id_matrix_0)
  # This seems to help avoid PSD error from the QP solver.
  block_kernel_matrix += 1.0 * torch.eye(n_way * n_support).expand(tasks_per_batch, n_way * n_support,
                                                                   n_way * n_support).cuda()

  support_labels_one_hot = one_hot(support_labels.view(tasks_per_batch * n_support),
                                   n_way)  # (tasks_per_batch * n_support, n_support)
  support_labels_one_hot = support_labels_one_hot.view(tasks_per_batch, n_support, n_way)
  support_labels_one_hot = support_labels_one_hot.reshape(tasks_per_batch, n_support * n_way)

  G = block_kernel_matrix
  e = -1.0 * support_labels_one_hot
  # print (G.size())
  # This part is for the inequality constraints:
  # \alpha^m_i <= C^m_i \forall m,i
  # where C^m_i = C if m  = y_i,
  # C^m_i = 0 if m != y_i.
  id_matrix_1 = torch.eye(n_way * n_support).expand(tasks_per_batch, n_way * n_support, n_way * n_support)
  C = Variable(id_matrix_1)
  h = Variable(C_reg * support_labels_one_hot)
  # print (C.size(), h.size())
  # This part is for the equality constraints:
  # \sum_m \alpha^m_i=0 \forall i
  id_matrix_2 = torch.eye(n_support).expand(tasks_per_batch, n_support, n_support).cuda()

  A = Variable(batched_kronecker(id_matrix_2, torch.ones(tasks_per_batch, 1, n_way).cuda()))
  b = Variable(torch.zeros(tasks_per_batch, n_support))
  # print (A.size(), b.size())
  if double_precision:
    G, e, C, h, A, b = [x.double().cuda() for x in [G, e, C, h, A, b]]
  else:
    G, e, C, h, A, b = [x.float().cuda() for x in [G, e, C, h, A, b]]

  # Solve the following QP to fit SVM:
  #        \hat z =   argmin_z 1/2 z^T G z + e^T z
  #                 subject to Cz <= h
  # We use detach() to prevent backpropagation to fixed variables.
  qp_sol = QPFunction(verbose=False, maxIter=maxIter)(G, e.detach(), C.detach(), h.detach(), A.detach(), b.detach())

  # Compute the classification score.
  compatibility = computeGramMatrix(support, query)
  compatibility = compatibility.float()
  compatibility = compatibility.unsqueeze(3).expand(tasks_per_batch, n_support, n_query, n_way)
  qp_sol = qp_sol.reshape(tasks_per_batch, n_support, n_way)
  logits = qp_sol.float().unsqueeze(2).expand(tasks_per_batch, n_support, n_query, n_way)
  logits = logits * compatibility
  logits = torch.sum(logits, 1)

  return logits


def MetaOptNetHead_SVM_WW(query, support, support_labels, n_way, n_shot, C_reg=0.00001, double_precision=False):
  """
  Fits the support set with multi-class SVM and
  returns the classification score on the query set.

  This is the multi-class SVM presented in:
  Support Vector Machines for Multi Class Pattern Recognition
  (Weston and Watkins, ESANN 1999).

  Parameters:
    query:  a (tasks_per_batch, n_query, d) Tensor.
    support:  a (tasks_per_batch, n_support, d) Tensor.
    support_labels: a (tasks_per_batch, n_support) Tensor.
    n_way: a scalar. Represents the number of classes in a few-shot classification task.
    n_shot: a scalar. Represents the number of support examples given per class.
    C_reg: a scalar. Represents the cost parameter C in SVM.
  Returns: a (tasks_per_batch, n_query, n_way) Tensor.
  """
  """
  Fits the support set with multi-class SVM and 
  returns the classification score on the query set.

  This is the multi-class SVM presented in:
  Support Vector Machines for Multi Class Pattern Recognition
  (Weston and Watkins, ESANN 1999).

  Parameters:
    query:  a (tasks_per_batch, n_query, d) Tensor.
    support:  a (tasks_per_batch, n_support, d) Tensor.
    support_labels: a (tasks_per_batch, n_support) Tensor.
    n_way: a scalar. Represents the number of classes in a few-shot classification task.
    n_shot: a scalar. Represents the number of support examples given per class.
    C_reg: a scalar. Represents the cost parameter C in SVM.
  Returns: a (tasks_per_batch, n_query, n_way) Tensor.
  """
  tasks_per_batch = query.size(0)
  n_support = support.size(1)
  n_query = query.size(1)

  assert (query.dim() == 3)
  assert (support.dim() == 3)
  assert (query.size(0) == support.size(0) and query.size(2) == support.size(2))
  assert (n_support == n_way * n_shot)  # n_support must equal to n_way * n_shot

  # In theory, \alpha is an (n_support, n_way) matrix
  # NOTE: In this implementation, we solve for a flattened vector of size (n_way*n_support)
  # In order to turn it into a matrix, you must first reshape it into an (n_way, n_support) matrix
  # then transpose it, resulting in (n_support, n_way) matrix
  kernel_matrix = computeGramMatrix(support, support) + torch.ones(tasks_per_batch, n_support, n_support).cuda()

  id_matrix_0 = torch.eye(n_way).expand(tasks_per_batch, n_way, n_way).cuda()
  block_kernel_matrix = batched_kronecker(id_matrix_0, kernel_matrix)

  kernel_matrix_mask_x = support_labels.reshape(tasks_per_batch, n_support, 1).expand(tasks_per_batch, n_support,
                                                                                      n_support)
  kernel_matrix_mask_y = support_labels.reshape(tasks_per_batch, 1, n_support).expand(tasks_per_batch, n_support,
                                                                                      n_support)
  kernel_matrix_mask = (kernel_matrix_mask_x == kernel_matrix_mask_y).float()

  block_kernel_matrix_inter = kernel_matrix_mask * kernel_matrix
  block_kernel_matrix += block_kernel_matrix_inter.repeat(1, n_way, n_way)

  kernel_matrix_mask_second_term = support_labels.reshape(tasks_per_batch, n_support, 1).expand(tasks_per_batch,
                                                                                                n_support,
                                                                                                n_support * n_way)
  kernel_matrix_mask_second_term = kernel_matrix_mask_second_term == torch.arange(n_way).long().repeat(
    n_support).reshape(n_support, n_way).transpose(1, 0).reshape(1, -1).repeat(n_support, 1).cuda()
  kernel_matrix_mask_second_term = kernel_matrix_mask_second_term.float()

  block_kernel_matrix -= (2.0 - 1e-4) * (kernel_matrix_mask_second_term * kernel_matrix.repeat(1, 1, n_way)).repeat(1,
                                                                                                                    n_way,
                                                                                                                    1)

  Y_support = one_hot(support_labels.view(tasks_per_batch * n_support), n_way)
  Y_support = Y_support.view(tasks_per_batch, n_support, n_way)
  Y_support = Y_support.transpose(1, 2)  # (tasks_per_batch, n_way, n_support)
  Y_support = Y_support.reshape(tasks_per_batch, n_way * n_support)

  G = block_kernel_matrix

  e = -2.0 * torch.ones(tasks_per_batch, n_way * n_support)
  id_matrix = torch.eye(n_way * n_support).expand(tasks_per_batch, n_way * n_support, n_way * n_support)

  C_mat = C_reg * torch.ones(tasks_per_batch, n_way * n_support).cuda() - C_reg * Y_support

  C = Variable(torch.cat((id_matrix, -id_matrix), 1))
  # C = Variable(torch.cat((id_matrix_masked, -id_matrix_masked), 1))
  zer = torch.zeros(tasks_per_batch, n_way * n_support).cuda()

  h = Variable(torch.cat((C_mat, zer), 1))

  dummy = Variable(torch.Tensor()).cuda()  # We want to ignore the equality constraint.

  if double_precision:
    G, e, C, h = [x.double().cuda() for x in [G, e, C, h]]
  else:
    G, e, C, h = [x.cuda() for x in [G, e, C, h]]

  # Solve the following QP to fit SVM:
  #        \hat z =   argmin_z 1/2 z^T G z + e^T z
  #                 subject to Cz <= h
  # We use detach() to prevent backpropagation to fixed variables.
  # qp_sol = QPFunction(verbose=False)(G, e.detach(), C.detach(), h.detach(), dummy.detach(), dummy.detach())
  qp_sol = QPFunction(verbose=False)(G, e, C, h, dummy.detach(), dummy.detach())

  # Compute the classification score.
  compatibility = computeGramMatrix(support, query) + torch.ones(tasks_per_batch, n_support, n_query).cuda()
  compatibility = compatibility.float()
  compatibility = compatibility.unsqueeze(1).expand(tasks_per_batch, n_way, n_support, n_query)
  qp_sol = qp_sol.float()
  qp_sol = qp_sol.reshape(tasks_per_batch, n_way, n_support)
  A_i = torch.sum(qp_sol, 1)  # (tasks_per_batch, n_support)
  A_i = A_i.unsqueeze(1).expand(tasks_per_batch, n_way, n_support)
  qp_sol = qp_sol.float().unsqueeze(3).expand(tasks_per_batch, n_way, n_support, n_query)
  Y_support_reshaped = Y_support.reshape(tasks_per_batch, n_way, n_support)
  Y_support_reshaped = A_i * Y_support_reshaped
  Y_support_reshaped = Y_support_reshaped.unsqueeze(3).expand(tasks_per_batch, n_way, n_support, n_query)
  logits = (Y_support_reshaped - qp_sol) * compatibility

  logits = torch.sum(logits, 2)

  return logits.transpose(1, 2)


class ClassificationHead(nn.Module):
  def __init__(self, base_learner='MetaOptNet', enable_scale=True):
    super(ClassificationHead, self).__init__()
    if ('SVM-CS' in base_learner):
      self.head = MetaOptNetHead_SVM_CS
    elif ('Ridge' in base_learner):
      self.head = MetaOptNetHead_Ridge
    elif ('R2D2' in base_learner):
      self.head = R2D2Head
    elif ('Proto' in base_learner):
      self.head = ProtoNetHead
    elif ('SVM-He' in base_learner):
      self.head = MetaOptNetHead_SVM_He
    elif ('SVM-WW' in base_learner):
      self.head = MetaOptNetHead_SVM_WW
    else:
      print("Cannot recognize the base learner type")
      assert (False)

    # Add a learnable scale
    self.enable_scale = enable_scale
    self.scale = nn.Parameter(torch.FloatTensor([1.0]))

  def forward(self, query, support, support_labels, n_way, n_shot, **kwargs):
    if self.enable_scale:
      return self.scale * self.head(query, support, support_labels, n_way, n_shot, **kwargs)
    else:
      return self.head(query, support, support_labels, n_way, n_shot, **kwargs)