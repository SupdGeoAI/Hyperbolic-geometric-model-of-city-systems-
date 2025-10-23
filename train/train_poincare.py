import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.dataset_poincare import DataReader, PoincareDataset
from utils.model_poincare import PoincareManifold, DistanceEnergyFunction, RiemannianSGD

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class PoincareEmbedding:
    def __init__(self,version,data_path,window_size,nneg,batch_size,emb_dimension,epochs,lr,epsilon,lr_exp_scheduler):
        self.data_path = data_path
        self.window_size = window_size
        self.data = DataReader(data_path,window_size)
        dataset = PoincareDataset(self.data,window_size,nneg)
        self.dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=True)
        self.emb_size = len(self.data.word2id)
        self.emb_dimension = emb_dimension
        self.version = version
        self.epochs = epochs
        self.lr = lr
        self.epsilon = epsilon
        self.lr_exp_scheduler = lr_exp_scheduler
        self.manifold = PoincareManifold(eps=epsilon)
        self.model = DistanceEnergyFunction(self.manifold,self.emb_dimension,self.emb_size)
        self.optimizer = RiemannianSGD(self.model.optim_params(),lr=self.lr)
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if self.use_cuda else "cpu")
        if self.use_cuda:
            self.model.to(self.device)
    
    def train(self):
        lr = self.lr
        save_log_path = '../log/v_' + str(self.version) + '/logs.txt'
        writer = SummaryWriter('../log/v_' + str(self.version) +'/tensorboard/')
        batch_step = 0
        f = open(save_log_path,'w')
        for epoch in range(self.epochs):
            print("\n\n\nEpoch: "+str(epoch+1))
            f.write("\n\n\nEpoch: "+str(epoch+1)+"\n")
            running_loss = 0.0
            for i,sample_batched in enumerate(tqdm(self.dataloader,desc="Batch Training")):
                batch_step += 1
                pos_u = sample_batched[0].to(self.device)
                pos_v = sample_batched[1].to(self.device)
                neg_v = sample_batched[2].to(self.device)
                pos_u = pos_u.unsqueeze(dim=1)
                pos_v = pos_v.unsqueeze(dim=1)
                inputs = torch.cat([pos_u,pos_v,neg_v],dim=1)
                targets = torch.zeros(inputs.shape[0],(inputs.shape[1]-1))
                targets[:,0] = 1.0
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                self.optimizer.zero_grad()
                preds = self.model(inputs)
                loss = self.model.loss(preds,targets)
                loss.backward()
                self.optimizer.step(lr=lr)
                running_loss = loss.item()
                if i>0 and i%100 == 0:
                    print(" Loss: " + str(running_loss))
                    f.write(" Loss: " + str(running_loss)+"\n")
                    writer.add_scalar(tag="Train Loss",scalar_value=running_loss,global_step=batch_step)
            self.model.save_embedding(epoch,self.data.id2word,self.version)
            lr = lr*self.lr_exp_scheduler
        f.close()
        writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--version',type=int)
    parser.add_argument('--data_path',type=str)
    parser.add_argument('--window_size',type=int)
    parser.add_argument('--nneg',type=int)
    parser.add_argument('--batch_size',type=int)
    parser.add_argument('--emb_dimension',type=int)
    parser.add_argument('--epochs',type=int)
    parser.add_argument('--lr',type=float)
    parser.add_argument('--lr_exp_scheduler',default=0.96,type=float)
    parser.add_argument('--seed',default=1,type=int)
    parser.add_argument('--epsilon',default=1e-5,type=float)
    args = parser.parse_args()

    argsDict = args.__dict__
    save_dir = '../log/v_'+str(args.version)
    os.mkdir(save_dir)
    save_setting_path = save_dir + '/settings.txt'
    with open(save_setting_path,'w') as f:
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : '+str(value)+'\n')
    f.close()
    set_seed(args.seed)
    model = PoincareEmbedding(version = args.version, data_path = args.data_path,
                              window_size = args.window_size, nneg = args.nneg,
                              batch_size = args.batch_size, emb_dimension = args.emb_dimension,
                              epochs = args.epochs, lr = args.lr,
                              epsilon = args.epsilon, lr_exp_scheduler = args.lr_exp_scheduler)
    model.train()