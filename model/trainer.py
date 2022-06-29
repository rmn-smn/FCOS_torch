import torch
import matplotlib.pyplot as plt

class Trainer():
    def __init__(self,
        model,
        train_loader,
        max_iter,
        max_lr = 0.01,
        weight_decay = 1e-4,
        checkpoint_path = '.',
        checkpoint_interval = 100,
        print_interval = 100
    ):
        
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        print(self.device)
        self.model = model.to(self.device)
        self.train_loader = self._infinite_loader(train_loader)
        self.last_iter = 0
        self.last_loss = 0
        self.max_iter = max_iter
        self.max_lr = max_lr
        self.weight_decay = weight_decay
        self.checkpoint_path = checkpoint_path
        self.checkpoint_interval = checkpoint_interval
        self.print_interval = print_interval
        self.optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            momentum=0.9,
            lr=max_lr,
            weight_decay=weight_decay,
        )

        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, 
            milestones=[int(0.6 * self.max_iter), int(0.9 * self.max_iter)]
        )

        self.stats = {
            'lrs': [],
            'train_loss': [],
            'train_reg_loss': [],
            'train_cls_loss': [],
            'train_ctr_loss': []
        }
    
    def _train(self,images, labels):
        self.model.train()

        losses = self.model(images,labels)
        total_loss = sum(losses.values())  

        # remove gradient from previous passes
        self.optimizer.zero_grad()

        # backprop
        total_loss.backward()

        self.optimizer.step()
        
        # Record stats & update learning rate
        self.stats['lrs'].append(self._get_lr())
        self.stats['train_loss'].append(total_loss.item())
        self.stats['train_reg_loss'].append(losses['reg_loss'].item())
        self.stats['train_ctr_loss'].append(losses['ctr_loss'].item())
        self.stats['train_cls_loss'].append(losses['cls_loss'].item())
        self.scheduler.step()

    def save_checkpoint(self,file_path):
        torch.save({
            'last_iter': self.last_iter,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'stats': self.stats,
            }, file_path
        )
        print('checkpoint saved')

    def load_checkpoint(self,file_path):
        checkpoint = torch.load(file_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.last_iter = checkpoint['last_iter']
        self.stats = checkpoint['stats']
        print('checkpoint loaded')

    def _infinite_loader(self,loader):
        """Get an infinite stream of batches from a data loader."""
        while True:
            yield from loader

    def _get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']
        
    def fit(self):
        print('running training')
        for i in range(self.last_iter+1, self.max_iter+1):
            (_,images,labels) = next(self.train_loader)
            images, labels = images.to(self.device), labels.to(self.device)   
            self._train(images, labels)
        
            if i%self.print_interval == 0:
                print(
                    'iteration {}/{}  training: total_loss={:.4f}, reg_loss={:.4f}, cls_loss={:.4f}, ctr_loss={:.4f}, lr={:.4f}'.format(
                        i,self.max_iter, self.stats['train_loss'][-1], 
                            self.stats['train_reg_loss'][-1], 
                            self.stats['train_cls_loss'][-1], 
                            self.stats['train_ctr_loss'][-1], self._get_lr()
                    )
                )
            self.last_iter = i
            
            if self.last_iter % self.checkpoint_interval == 0:
                self.save_checkpoint(self.checkpoint_path)
                
        plt.title("Training loss history")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.plot(self.stats['train_reg_loss'],label='bbox')
        plt.plot(self.stats['train_cls_loss'],label='class')
        plt.plot(self.stats['train_ctr_loss'],label='centerness')
        plt.legend()
        plt.show()
