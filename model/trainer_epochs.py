import torch

class Trainer():
    def __init__(self,
        model,
        train_loader,
        val_loader,
        epochs,
        max_lr = 0.01,
        weight_decay = 1e-4,
        checkpoint_path = '.'
    ):
        
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        print(self.device)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.last_epoch = 0
        self.last_loss = 0
        self.max_iters = len(self.train_loader)*epochs
        self.max_lr = max_lr
        self.weight_decay = weight_decay
        self.checkpoint_path = checkpoint_path
        self.optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            momentum=0.9,
            lr=max_lr,
            weight_decay=weight_decay,
        )
        # torch.optim.Adam(
        #     model.parameters(), max_lr, weight_decay=weight_decay
        # )
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, 
            milestones=[int(0.6 * self.max_iters), int(0.9 * self.max_iters)]
        )
        #torch.optim.lr_scheduler.OneCycleLR(
        #    self.optimizer, max_lr, epochs=epochs, 
        #    steps_per_epoch=len(train_loader)
        #)

        self.stats = {
            'lrs': [],
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }

    
    def _train(self):
        self.model.train()
        for i,(_,images,labels) in enumerate(self.train_loader):

            images, labels = images.to(self.device), labels.to(self.device)
            losses = self.model(images,labels)
            total_loss = sum(losses.values())  

            if i%20 == 0:
                print(
                    'iteration {}/{}  validation: total_loss={:.4f}, reg_loss={:.4f}, cls_loss={:.4f}, ctr_loss={:.4f}, lr={:.4f}'.format(
                        i,len(self.train_loader)-1, total_loss, \
                            losses['reg_loss'], losses['cls_loss'], \
                                losses['ctr_loss'], self.get_lr()
                    )
                )

            # remove gradient from previous passes
            self.optimizer.zero_grad()

            # backprop
            total_loss.backward()

            self.optimizer.step()
            
            # Record stats & update learning rate
            self.stats['lrs'].append(self.get_lr())
            self.stats['train_loss'].append(total_loss)
            self.scheduler.step()
        
    @torch.no_grad()
    def _validate(self):

        self.model.eval()
        for i,(path,images,labels) in enumerate(self.val_loader):

            images, labels = images.to(self.device), labels.to(self.device)
            losses = self.model(images,labels)
            total_loss = sum(losses.values())

            if i%(10-1) == 0:
              print(
                  'iteration {}/{}  validation: total_loss={:.4f}, reg_loss={:.4f}, cls_loss={:.4f}, ctr_loss={:.4f}'.format(
                     i,len(self.val_loader)-1, total_loss, losses['reg_loss'], losses['cls_loss'], losses['ctr_loss']))                      

            self.stats['val_loss'].append(total_loss)

    def save_checkpoint(self,epoch,file_path):
        torch.save({
            'epoch': epoch,
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
        self.last_epoch = checkpoint['epoch']
        self.stats = checkpoint['stats']
        print('checkpoint loaded')

    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']
        
    def fit(self):
        print('running training')
        for epoch in range(self.last_epoch, self.epochs):
            print('current epoch: {}'.format(epoch))
            
            self._train()
            #print()
            #val_loss = self._validate()

            self.last_epoch = epoch

            if epoch % 10 == 0:
                self.save_checkpoint(epoch,self.checkpoint_path)
                

            #print()
            #print(
            #    "Summary: Epoch [{}], , train_loss: {:.4f}"
            #    .format(epoch, self.stats['train_loss']))
            #self.stats['lrs'][-1]