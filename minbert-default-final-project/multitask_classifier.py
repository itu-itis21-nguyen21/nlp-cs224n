import time, random, numpy as np, argparse, sys, re, os
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler

from bert import BertModel
from optimizer import AdamW
from tqdm import tqdm
#from pcgrad import PCGrad

from datasets import SentenceClassificationDataset, SentencePairDataset, \
    load_multitask_data, load_multitask_test_data

from evaluation import model_eval_sst, model_eval_para, model_eval_sts, model_eval_multitask, test_model_multitask

print("v19")

TQDM_DISABLE=False

# fix the random seed
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


BERT_HIDDEN_SIZE = 768
N_SENTIMENT_CLASSES = 5


class MultitaskBERT(nn.Module):
    '''
    This module should use BERT for 3 tasks:

    - Sentiment classification (predict_sentiment)
    - Paraphrase detection (predict_paraphrase)
    - Semantic Textual Similarity (predict_similarity)
    '''
    def __init__(self, config):
        super(MultitaskBERT, self).__init__()
        # You will want to add layers here to perform the downstream tasks.
        # Pretrain mode does not require updating bert paramters.
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        for param in self.bert.parameters():
            if config.option == 'pretrain':
                param.requires_grad = False
            elif config.option == 'finetune':
                param.requires_grad = True
        ### TODO
        # STS classifier
        self.dropout1 = nn.Dropout(p=0.3)        # twitch-able
        self.sst_out = nn.Linear(BERT_HIDDEN_SIZE, N_SENTIMENT_CLASSES)

        # Paraphrasing classifier
        self.dropout2 = nn.Dropout(p=0.3)        # twitch-able
        self.para_out = nn.Linear(2 * BERT_HIDDEN_SIZE, 1)

        # STS classifier
        self.dropout3 = nn.Dropout(p=0.3)        # twitch-able
        self.sts_out = nn.Linear(2 * BERT_HIDDEN_SIZE, 1)

    def forward(self, input_ids, attention_mask):
        'Takes a batch of sentences and produces embeddings for them.'
        # The final BERT embedding is the hidden state of [CLS] token (the first token)
        # Here, you can start by just returning the embeddings straight from BERT.
        # When thinking of improvements, you can later try modifying this
        # (e.g., by adding other layers).
        ### TODO
        # return the [CLS] embedding (pooler_output) and all words' embeddings (last_hidden_state),
        # all contained within the output as we call bert()
        return self.bert(input_ids, attention_mask)


    def predict_sentiment(self, input_ids, attention_mask):
        '''Given a batch of sentences, outputs logits for classifying sentiment.
        There are 5 sentiment classes:
        (0 - negative, 1- somewhat negative, 2- neutral, 3- somewhat positive, 4- positive)
        Thus, your output should contain 5 logits for each sentence.
        '''
        ### TODO
        # Use pooler_output as sentence embedding here
        sentence_embedding = self.forward(input_ids, attention_mask)["pooler_output"]
        output = self.dropout1(sentence_embedding)
        output = self.sst_out(output)
        return F.softmax(output, dim=-1)


    def predict_paraphrase(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit for predicting whether they are paraphrases.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation, and handled as a logit by the appropriate loss function.
        '''
        ### TODO
        # Retrieve pooler_output of 2 sentences
        sent1 = self.forward(input_ids_1, attention_mask_1)["pooler_output"]
        sent2 = self.forward(input_ids_2, attention_mask_2)["pooler_output"]
        # Add their embeddings
        assert sent1.shape == sent2.shape
        sent = torch.cat((sent1, sent2), dim=1)
        # Feed through a fully connected layer
        output = self.dropout2(sent)
        return self.para_out(output)

    def predict_similarity(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit corresponding to how similar they are.
        Note that your output should be unnormalized (a logit).
        '''
        ### TODO
        # Retrieve last_hidden_state of 2 sentences
        sent1 = self.forward(input_ids_1, attention_mask_1)["last_hidden_state"]
        sent2 = self.forward(input_ids_2, attention_mask_2)["last_hidden_state"]

        # Remove the paddings by multiplying embeddings with masks
        attention_mask_1 = attention_mask_1.unsqueeze(-1).expand(sent1.shape).float()
        attention_mask_2 = attention_mask_2.unsqueeze(-1).expand(sent2.shape).float()
        sent1 *= attention_mask_1
        sent2 *= attention_mask_2

        # Mean pooling
        sum1 = torch.sum(sent1, dim=1)
        sum2 = torch.sum(sent2, dim=1)
        count1 = torch.clamp(attention_mask_1.sum(1), min=1e-9)
        count2 = torch.clamp(attention_mask_2.sum(1), min=1e-9)
        mean_pooled1 = sum1 / count1
        mean_pooled2 = sum2 / count2

        # Concat their embeddings
        assert(mean_pooled1.shape == mean_pooled2.shape)
        mean_pooled = torch.cat((mean_pooled1, mean_pooled2), dim=-1)

        # Feed through a fully connected layer
        output = self.dropout3(mean_pooled)
        return self.sts_out(output)


def save_model(model, optimizer, args, config, filepath):
    save_info = {
        'model': model.state_dict(),
        'args': args,
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")


## Train on sst, quora and sts datasets
def train_multitask(args):
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    # Load data
    # Create the data and its corresponding datasets and dataloader
    sst_train_data, num_labels,para_train_data, sts_train_data = load_multitask_data(args.sst_train,args.para_train,args.sts_train, split ='train')
    sst_dev_data, num_labels,para_dev_data, sts_dev_data = load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev, split ='train')

    sst_train_data = SentenceClassificationDataset(sst_train_data, args)
    sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)
    para_train_data = SentencePairDataset(para_train_data, args)
    para_dev_data = SentencePairDataset(para_dev_data, args)
    sts_train_data = SentencePairDataset(sts_train_data, args)
    sts_dev_data = SentencePairDataset(sts_dev_data, args)

    # We want to train our model on 3 datasets with the SAME number of samples
    # Because the STS dataset is the smallest among three, we randomly sample
    # from the SST dataset and Quora dataset to have 3 datasets with smiliar
    # number of samples.
    sst_train_sampler = RandomSampler(sst_train_data, replacement=True, num_samples=(len(sts_train_data)))
    para_train_sampler = RandomSampler(para_train_data, replacement=True, num_samples=(len(sts_train_data)))

    sst_train_dataloader = DataLoader(sst_train_data, sampler=sst_train_sampler, batch_size=args.batch_size,
                                      collate_fn=sst_train_data.collate_fn)
    sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sst_dev_data.collate_fn)
    para_train_dataloader = DataLoader(para_train_data, sampler=para_train_sampler, batch_size=args.batch_size,
                                      collate_fn=para_train_data.collate_fn)
    para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=para_dev_data.collate_fn)
    sts_train_dataloader = DataLoader(sts_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=sts_train_data.collate_fn)
    sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sts_dev_data.collate_fn)

    # Init model
    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'num_labels': num_labels,
              'hidden_size': 768,
              'data_dir': '.',
              'option': args.option}

    config = SimpleNamespace(**config)
    model = MultitaskBERT(config)
    model = model.to(device)

    lr = args.lr
    optimizer = AdamW(model.parameters(), lr=lr)
    best_dev_acc = 0
    # Run for the specified number of epochs
    for epoch in range(args.epochs):
        model.train()
        sst_train_loss = 0
        para_train_loss = 0
        sts_train_loss = 0
        train_loss = 0
        num_batches = 0
        for batch_sst, batch_para, batch_sts in tqdm(zip(sst_train_dataloader, para_train_dataloader, sts_train_dataloader), \
                                                        desc=f'train-{epoch}', disable=TQDM_DISABLE):
            # processing data
            b_sst_ids, b_sst_mask, b_sst_labels = (batch_sst['token_ids'], batch_sst['attention_mask'], batch_sst['labels'])
            b_sst_ids = b_sst_ids.to(device)
            b_sst_mask = b_sst_mask.to(device)
            b_sst_labels = b_sst_labels.to(device)

            b_para_ids_1, b_para_ids_2 = (batch_para['token_ids_1'], batch_para["token_ids_2"])
            b_para_mask_1, b_para_mask_2, b_para_labels = (batch_para['attention_mask_1'], batch_para['attention_mask_2'], batch_para['labels'])
            b_para_ids_1 = b_para_ids_1.to(device)
            b_para_ids_2 = b_para_ids_2.to(device)
            b_para_mask_1 = b_para_mask_1.to(device)
            b_para_mask_2 = b_para_mask_2.to(device)
            b_para_labels = b_para_labels.to(device)

            b_sts_ids_1, b_sts_ids_2 = (batch_sts['token_ids_1'], batch_sts["token_ids_2"])
            b_sts_mask_1, b_sts_mask_2, b_sts_labels = (batch_sts['attention_mask_1'], batch_sts['attention_mask_2'], batch_sts['labels'])
            b_sts_ids_1 = b_sts_ids_1.to(device)
            b_sts_ids_2 = b_sts_ids_2.to(device)
            b_sts_mask_1 = b_sts_mask_1.to(device)
            b_sts_mask_2 = b_sts_mask_2.to(device)
            b_sts_labels = b_sts_labels.to(device)

            optimizer.zero_grad()

            # predicting
            logits_sst = model.predict_sentiment(b_sst_ids, b_sst_mask)
            logits_para = model.predict_paraphrase(b_para_ids_1, b_para_mask_1, \
                                                  b_para_ids_2, b_para_mask_2)
            logits_sts = model.predict_similarity(b_sts_ids_1, b_sts_mask_1, \
                                                  b_sts_ids_2, b_sts_mask_2)
            
            # calculating loss
            loss_sst = F.cross_entropy(logits_sst, b_sst_labels.view(-1), reduction='sum') / args.batch_size
            loss_para = F.binary_cross_entropy_with_logits(logits_para, b_para_labels.float().unsqueeze(-1), reduction='sum') / args.batch_size
            loss_sts = F.mse_loss(logits_sts, b_sts_labels.float().unsqueeze(-1), reduction='sum') / args.batch_size
            total_losses = 0.45 * loss_sst + 0.45 * loss_para + 0.01 * loss_sts

            total_losses.backward()
            optimizer.step()

            sst_train_loss += loss_sst.item()
            para_train_loss += loss_para.item()
            sts_train_loss += loss_sts.item()
            train_loss += total_losses.item()
            num_batches += 1

        sst_train_loss /= num_batches
        para_train_loss /= num_batches
        sts_train_loss /= num_batches
        train_loss = train_loss / (num_batches)

        train_para_acc, _ , _ , \
        train_sst_acc, _ , _ , \
        train_sts_corr, _ , _ = model_eval_multitask(sst_train_dataloader, para_train_dataloader, sts_train_dataloader, model, device)
        dev_para_acc, _ , _ , \
        dev_sst_acc, _ , _ , \
        dev_sts_corr, _ , _ = model_eval_multitask(sst_dev_dataloader, para_dev_dataloader, sts_dev_dataloader, model, device)

        # dev_acc = mean of 3 task accuracies
        train_acc = (train_para_acc + train_sst_acc + train_sts_corr) / 3
        dev_acc = (dev_para_acc + dev_sst_acc + dev_sts_corr) / 3

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            save_model(model, optimizer, args, config, args.filepath)

        print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, train mean acc:: {train_acc :.3f}, dev mean acc:: {dev_acc :.3f}")
        print(f"SST loss:: {sst_train_loss :.3f}, Para loss:: {para_train_loss :.3f}, STS loss:: {sts_train_loss :.3f}")


def test_model(args):
    with torch.no_grad():
        device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
        saved = torch.load(args.filepath, map_location=torch.device("cpu"))         
        config = saved['model_config']

        model = MultitaskBERT(config)
        model.load_state_dict(saved['model'])
        model = model.to(device)
        print(f"Loaded model to test from {args.filepath}")

        test_model_multitask(args, model, device)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sst_train", type=str, default="data/ids-sst-train.csv")
    parser.add_argument("--sst_dev", type=str, default="data/ids-sst-dev.csv")
    parser.add_argument("--sst_test", type=str, default="data/ids-sst-test-student.csv")

    parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
    parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
    parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")

    parser.add_argument("--sts_train", type=str, default="data/sts-train.csv")
    parser.add_argument("--sts_dev", type=str, default="data/sts-dev.csv")
    parser.add_argument("--sts_test", type=str, default="data/sts-test-student.csv")

    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--option", type=str,
                        help='pretrain: the BERT parameters are frozen; finetune: BERT parameters are updated',
                        choices=('pretrain', 'finetune'), default="pretrain")
    parser.add_argument("--use_gpu", action='store_true')

    parser.add_argument("--sst_dev_out", type=str, default="predictions/sst-dev-output.csv")
    parser.add_argument("--sst_test_out", type=str, default="predictions/sst-test-output.csv")

    parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
    parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")

    parser.add_argument("--sts_dev_out", type=str, default="predictions/sts-dev-output.csv")
    parser.add_argument("--sts_test_out", type=str, default="predictions/sts-test-output.csv")

    # hyper parameters
    parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate, default lr for 'pretrain': 1e-3, 'finetune': 1e-5",
                        default=1e-5)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    args.filepath = f'{args.option}-{args.epochs}-{args.lr}-multitask.pt' # save path
    seed_everything(args.seed)  # fix the seed for reproducibility
    train_multitask(args)
    #test_model(args)
