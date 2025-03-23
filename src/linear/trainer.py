from abc import ABC, abstractmethod


class TrainingStrategy(ABC):
    def __init__(self, optimizer, loss, regularizer):
        self.optimizer = optimizer
        self.loss = loss
        self.regularizer = regularizer
    
    @abstractmethod
    def train_epoch(self, mini_batches, model):
        pass
    
    def compute_gradients(self, y_pred, y_true, x, model):
        penalty_term = 0
        if self.regularizer is not None:
            penalty_term = self.regularizer.derivative(model.weight_matrix)
            
        delta_gradient = self.loss.derivative(y_pred, y_true)
        gradient = (delta_gradient * x + penalty_term)
        
        return delta_gradient, gradient


class SGDTrainingStrategy(TrainingStrategy):
    def train_epoch(self, mini_batches, model):
        errors = []
        
        for i, (batch_x, batch_y) in enumerate(mini_batches):
            for j in range(batch_x.shape[0]):
                y_pred = model.predict(batch_x[j])
                errors.append(self.loss(y_pred, batch_y[j]))
                
                delta_gradient, gradient = self.compute_gradients(y_pred, batch_y[j], batch_x[j], model)
                self.update_parameters(model, delta_gradient, gradient, i)
                
        return sum(errors) / len(errors) if errors else 0
    
    def update_parameters(self, model, delta_gradient, gradient, batch_idx):
        if self.optimizer.name in ["adam", "sgd"]:
            model.weight_matrix, model.bias = self.optimizer(
                model.weight_matrix, model.bias, gradient, delta_gradient, batch_idx)

        else:
            model.weight_matrix, model.bias = self.optimizer(
                model.weight_matrix, model.bias, gradient, delta_gradient)


class MiniBatchTrainingStrategy(TrainingStrategy):
    def train_epoch(self, mini_batches, model):
        errors = []
        
        for i, (batch_x, batch_y) in enumerate(mini_batches):
            batch_deltas = []
            batch_gradients = []
            
            for j in range(batch_x.shape[0]):
                y_pred = model.predict(batch_x[j])
                errors.append(self.loss(y_pred, batch_y[j]))
                
                delta_gradient, gradient = self.compute_gradients(y_pred, batch_y[j], batch_x[j], model)
                batch_deltas.append(delta_gradient)
                batch_gradients.append(gradient)

            avg_delta = sum(batch_deltas) / len(batch_deltas)
            avg_gradient = sum(batch_gradients) / len(batch_gradients)
            
            self.update_parameters(model, avg_delta, avg_gradient, i)
                
        return sum(errors) / len(errors) if errors else 0
    
    def update_parameters(self, model, delta_gradient, gradient, batch_idx):
        if self.optimizer.name in ["adam", "sgd"]:
            model.weight_matrix, model.bias = self.optimizer(
                model.weight_matrix, model.bias, gradient, delta_gradient, batch_idx)

        else:
            model.weight_matrix, model.bias = self.optimizer(
                model.weight_matrix, model.bias, gradient, delta_gradient)