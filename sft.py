def fine_tune():
    global best_val_loss, counter
    model.train()
    for epoch in range(cfg.epochs):
        total_train_loss = 0
        total_train_acc = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits.view(-1, vocab_size), y.view(-1))
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            total_train_acc += calc_accuracy(logits.view(-1, vocab_size), y.view(-1))

        total_train_loss /= len(train_loader)
        total_train_acc /= len(train_loader)
        print(f"Fine-tuning Epoch {epoch}: Loss: {total_train_loss:.4f}, Acc: {total_train_acc:.4f}")

        model.eval()
        total_val_loss = 0
        total_val_acc = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                loss = loss_fn(logits.view(-1, vocab_size), y.view(-1))
                total_val_loss += loss.item()
                total_val_acc += calc_accuracy(logits.view(-1, vocab_size), y.view(-1))

        total_val_loss /= len(val_loader)
        total_val_acc /= len(val_loader)

        print(f"Validation Epoch {epoch}: Loss: {total_val_loss:.4f}, Acc: {total_val_acc:.4f}")

        # 保存最佳模型
        if total_val_loss < best_val_loss - 1e-5:
            best_val_loss = total_val_loss
            counter = 0
            torch.save(model.state_dict(), cfg.save_path)
        else:
            counter += 1
        if counter >= 3:  # Early stop
            print(f"Early stopping triggered at epoch {epoch}")
            break

        save_checkpoint(model, optimizer, epoch, cfg.checkpoint_path)

    print(f"微调完成，模型权重已保存到 {cfg.save_path}")
