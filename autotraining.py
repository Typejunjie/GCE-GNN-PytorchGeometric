import numpy as np
from tqdm import tqdm

class autotraining(object):
    def __init__(self, model, loss_fun, optimizer, writer=None, topk=20, scheduler=None, device='cpu', test_flag=False) -> None:
        self.model = model
        self.loss_fun = loss_fun
        self.optim = optimizer
        self.device = device
        self.writer = writer
        self.scheduler = scheduler
        self.topk = topk
        self.test_flag = test_flag
        self.hit = []
        self.mrr = []
        self.eval_loss = [] # 每次调用train_eval得到的平均损失值
        self.train_loss = [] # 每次调用train最后得到的损失值
    
    def train(self, dataloader, i):
        self.model.train()
        train_loss_his = []
        length = len(dataloader)
        
        for epoch, data in enumerate(tqdm(dataloader, colour='green', desc=f'Epoch {i + 1}', leave=False)):
            self.optim.zero_grad()
            predict = self.model(data.to(self.device))
            loss = self.loss_fun(predict, data.y - 1)
            loss.backward()
            self.optim.step()
            
            if epoch % 10 == 0:
                if self.test_flag == False:
                    self.writer.add_scalar('loss/train_batch_loss', loss.item(), i * length + epoch)
                train_loss_his.append(loss.item())

        return train_loss_his

    def fit(self, dataloader, evaldataloader=None, epochs=10, log_parameter=True, eval=False):
        loss_history = []

        for epoch in tqdm(range(epochs), colour='green', desc='Training model', leave=False):

            if self.scheduler != None:
                self.scheduler.step()

            loss_cache = self.train(dataloader, epoch)

            if loss_cache != None:
                loss_history.extend(loss_cache)
                mean_loss = np.mean(np.array(loss_cache))
                if self.test_flag == False:
                    self.writer.add_scalar('loss/train_loss', mean_loss, epoch)
                self.train_loss.append(mean_loss)

            if log_parameter:
                print(f'Epoch {epoch + 1} Loss {np.mean(np.array(loss_cache))}')
            
            if eval:
                eval_loss = self.evaluate(evaldataloader, epoch + 1, log_parameter=log_parameter)
                if self.test_flag == False:
                    self.writer.add_scalar('loss/test_loss', eval_loss, epoch)
                self.eval_loss.append(eval_loss)
                
                if log_parameter:
                    print(f'evaluate loss {eval_loss}')
        
        self.train_loss_history = loss_history

    def evaluate(self, dataloader, i, log_parameter=True):
        self.model.eval()
        hit, mrr = [], []
        loss_list = []

        for data in tqdm(dataloader, colour='green', desc='Estimating', leave=False):
            predict = self.model(data.to(self.device))
            targets = data.y - 1
            loss = self.loss_fun(predict, targets)
            loss_list.append(loss.item())

            topk_index = predict.topk(self.topk)[1]
            for pre, target in zip(topk_index.detach().cpu().numpy(), targets.detach().cpu().numpy()):
                hit.append(np.isin(target, pre))
                if len(np.where(pre == target)[0]) == 0:
                    mrr.append(0)
                else:
                    mrr.append(1 / (np.where(pre == target)[0][0] + 1))

        hit = np.mean(hit) * 100
        mrr = np.mean(mrr) * 100
        if self.test_flag == False:
            self.writer.add_scalar('index/hit@20', hit, i)
            self.writer.add_scalar('index/mrr@20', mrr, i)
        self.hit.append(hit)
        self.mrr.append(mrr)

        if log_parameter:
            print(f'hit {hit}, mrr@{self.topk} {mrr}')

        return np.mean(np.array(loss_list))



            

            

