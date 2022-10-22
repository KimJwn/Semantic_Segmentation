import torch
from pytorch_lightning import Trainer #, LightningDataModule, LightningModule

import module

# Weights & Biases
from pytorch_lightning.loggers import WandbLogger
import wandb



def main():
    wandb.finish()
    wandb.login()
    wandb.init(project="2018125019_KimJiWon_pytorch-lightning-Cifar10", entity="merily", name="cifar10_lenet")

    # 딥러닝 모델을 설계할 때 활용하는 장비 확인
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print('*************************Using PyTorch version:', torch.__version__, ' Device:', device)

    #GPU 갯수 확인
    AVAIL_GPUS = min(1, torch.cuda.device_count())
    print(AVAIL_GPUS)

    # setup data
    cifar = module.DataModule(batch_size=256)
    
    # setup model
    model = module.LitCF(n_classes=10, lr=0.08)

    # setup trainer
    wandb_logger = WandbLogger(entity='merily', project='2018125019_KimJiWon_pytorch-lightning-Cifar10')
    trainer = Trainer(accelerator='gpu', strategy='dp',
        logger=wandb_logger,    # W&B integration
        gpus=1,                # use all GPU's
        max_epochs=20,            # number of epochs
        )

    trainer.fit(model=model, datamodule=cifar)
    trainer.test(model = model, datamodule=cifar)

    wandb.finish()
    return 0
#





if __name__ == "__main__":
    # main()
    print()
