# Used to improve the performance of doing paraphrasing task
def train_paraphrase(args):
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    # Load data
    # Create the data and its corresponding datasets and dataloader
    sst_train_data, num_labels,para_train_data, sts_train_data = load_multitask_data(args.sst_train,args.para_train,args.sts_train, split ='train')
    sst_dev_data, num_labels,para_dev_data, sts_dev_data = load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev, split ='train')

    para_train_data = SentencePairDataset(para_train_data, args)
    para_dev_data = SentencePairDataset(para_dev_data, args)

    # sample 1/10 of each dataset for training in each epoch
    para_train_sampler = RandomSampler(para_train_data, replacement=True, num_samples=(len(para_train_data)//10))

    para_train_dataloader = DataLoader(para_train_data, sampler=para_train_sampler, batch_size=args.batch_size,
                                      collate_fn=para_train_data.collate_fn)
    para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=para_dev_data.collate_fn)

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
        train_loss = 0
        num_batches = 0
        for batch_para in tqdm(para_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
            # processing data
            b_para_ids_1, b_para_ids_2 = (batch_para['token_ids_1'], batch_para["token_ids_2"])
            b_para_mask_1, b_para_mask_2, b_para_labels = (batch_para['attention_mask_1'], batch_para['attention_mask_2'], batch_para['labels'])
            b_para_ids_1 = b_para_ids_1.to(device)
            b_para_ids_2 = b_para_ids_2.to(device)
            b_para_mask_1 = b_para_mask_1.to(device)
            b_para_mask_2 = b_para_mask_2.to(device)
            b_para_labels = b_para_labels.to(device)

            optimizer.zero_grad()
            logits_para = model.predict_paraphrase(b_para_ids_1, b_para_mask_1, \
                                                  b_para_ids_2, b_para_mask_2)
            
            # calculating loss
            loss = F.binary_cross_entropy_with_logits(logits_para, b_para_labels.float().unsqueeze(-1), reduction='sum') / args.batch_size

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        train_loss = train_loss / (num_batches)

        train_acc, _ , _ = model_eval_para(para_train_dataloader, model, device)
        dev_acc, _ , _ = model_eval_para(para_dev_dataloader, model, device)

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            save_model(model, optimizer, args, config, args.filepath)

        print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, train mean acc:: {train_acc :.3f}, dev mean acc:: {dev_acc :.3f}")

# Used to improve the performance of doing similarity task
def train_similarity(args):
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    # Load data
    # Create the data and its corresponding datasets and dataloader
    sst_train_data, num_labels,para_train_data, sts_train_data = load_multitask_data(args.sst_train,args.para_train,args.sts_train, split ='train')
    sst_dev_data, num_labels,para_dev_data, sts_dev_data = load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev, split ='train')

    sts_train_data = SentencePairDataset(sts_train_data, args)
    sts_dev_data = SentencePairDataset(sts_dev_data, args)

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
    best_dev_corr = 0
    # Run for the specified number of epochs
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0
        for batch_sts in tqdm(sts_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
            # processing data
            b_sts_ids_1, b_sts_ids_2 = (batch_sts['token_ids_1'], batch_sts["token_ids_2"])
            b_sts_mask_1, b_sts_mask_2, b_sts_labels = (batch_sts['attention_mask_1'], batch_sts['attention_mask_2'], batch_sts['labels'])
            b_sts_ids_1 = b_sts_ids_1.to(device)
            b_sts_ids_2 = b_sts_ids_2.to(device)
            b_sts_mask_1 = b_sts_mask_1.to(device)
            b_sts_mask_2 = b_sts_mask_2.to(device)
            b_sts_labels = b_sts_labels.to(device)

            optimizer.zero_grad()

            # predicting
            logits_sts = model.predict_similarity(b_sts_ids_1, b_sts_mask_1, \
                                                  b_sts_ids_2, b_sts_mask_2)
            
            # calculating loss
            loss = F.mse_loss(logits_sts, b_sts_labels.float().unsqueeze(-1), reduction='sum') / args.batch_size

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        train_loss = train_loss / (num_batches)

        train_corr, _ , _ = model_eval_sts(sts_train_dataloader, model, device)
        dev_corr, _ , _ = model_eval_sts(sts_dev_dataloader, model, device)

        if dev_corr > best_dev_corr:
            best_dev_corr = dev_corr
            save_model(model, optimizer, args, config, args.filepath)

        print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, train corr:: {train_corr :.3f}, dev corr:: {dev_corr :.3f}")

## Training version with PCGrad (gradient surgery)
## https://github.com/WeiChengTseng/Pytorch-PCGrad
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
    optimizer = PCGrad(AdamW(model.parameters(), lr=lr))
    best_dev_acc = 0
    # Run for the specified number of epochs
    for epoch in range(args.epochs):
        model.train()
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
            total_losses = loss_sst + loss_para + loss_sts
            losses = [loss_sst, loss_para, loss_sts]

            optimizer.pc_backward(losses)
            optimizer.step()

            train_loss += total_losses.item()
            num_batches += 1

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


## Vanilla training function
def train_multitask(args):
loss_sst = F.cross_entropy(logits_sst, b_sst_labels.view(-1), reduction='sum') / args.batch_size
loss_para = F.binary_cross_entropy_with_logits(logits_para, b_para_labels.float().unsqueeze(-1), reduction='sum') / args.batch_size
loss_sts = F.mse_loss(logits_sts, b_sts_labels.float().unsqueeze(-1), reduction='sum') / args.batch_size
total_losses = loss_sst + loss_para + loss_sts

total_losses.backward()
optimizer.step()


## Optimizng for worst-case task loss
loss_sst = F.cross_entropy(logits_sst, b_sst_labels.view(-1), reduction='sum') / args.batch_size
loss_para = F.binary_cross_entropy_with_logits(logits_para, b_para_labels.float().unsqueeze(-1), reduction='sum') / args.batch_size
loss_sts = F.mse_loss(logits_sts, b_sts_labels.float().unsqueeze(-1), reduction='sum') / args.batch_size
losses = [loss_sst, loss_para, loss_sts]
total_losses = loss_sst + loss_para + loss_sts

# optimize for worst-case task loss
max_loss = losses.index(max(losses))
if max_loss == 0:
    loss_sst.backward()
elif max_loss == 1:
    loss_para.backward()
else:
    loss_sts.backward()

optimizer.step()

# Optimize on only SST task
loss_sst = F.cross_entropy(logits_sst, b_sst_labels.view(-1), reduction='sum') / args.batch_size
loss_para = F.binary_cross_entropy_with_logits(logits_para, b_para_labels.float().unsqueeze(-1), reduction='sum') / args.batch_size
loss_sts = F.mse_loss(logits_sts, b_sts_labels.float().unsqueeze(-1), reduction='sum') / args.batch_size
total_losses = loss_sst + loss_para + loss_sts

loss_sst.backward()
optimizer.step()

 # objective function with weights assigned to each task
loss_sst = F.cross_entropy(logits_sst, b_sst_labels.view(-1), reduction='sum') / args.batch_size
loss_para = F.binary_cross_entropy_with_logits(logits_para, b_para_labels.float().unsqueeze(-1), reduction='sum') / args.batch_size
loss_sts = F.mse_loss(logits_sts, b_sts_labels.float().unsqueeze(-1), reduction='sum') / args.batch_size
total_losses = 0.45 * loss_sst + 0.45 * loss_para + 0.01 * loss_sts

total_losses.backward()
optimizer.step()