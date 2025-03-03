from model import *

# 运行 Optuna 超参数优化
best_params = load_best_hyperparams()
if best_params:
    print("📌 载入的最优超参数：", best_params)
    
else:
    print("🎯 喵呜~ 开始寻找最棒的超参数组合！")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)

    # 获取最佳超参数并保存
    best_params = study.best_params
    print(f"🏆 喵喵找到最佳参数啦！{best_params}")
    save_best_hyperparams(best_params)


# 训练最终模型
final_model = RegressionModel(input_size=X.shape[1], hidden_size=best_params["hidden_size"], num_layers=best_params["num_layers"]).to(device)
final_optimizer = optim.Adam(final_model.parameters(), lr=best_params["lr"])
final_criterion = nn.MSELoss()
train_loader = data.DataLoader(train_dataset, batch_size=best_params["batch_size"], shuffle=True)
val_loader = data.DataLoader(val_dataset, batch_size=best_params["batch_size"], shuffle=False)

best_val_loss = float("inf")
early_stopping = EarlyStopping(patience=10)

print("🚀 开始最终训练喵！")
for epoch in range(100):
    final_model.train()
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        final_optimizer.zero_grad()
        outputs = final_model(batch_X)
        loss = final_criterion(outputs, batch_y)
        loss.backward()
        final_optimizer.step()

    final_model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = final_model(batch_X)
            val_loss += final_criterion(outputs, batch_y).item()
    val_loss /= len(val_loader)

    print(f"📊 最终模型 - Epoch {epoch+1}: 验证损失 {val_loss:.6f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(final_model.state_dict(), "best_model.pth")
        print("💾 喵喵存好了最棒的模型！")

    if early_stopping.step(val_loss):
        print("💤 早停触发！喵不练了！")
        break

print(f"🎉 最终训练完成！最佳验证损失: {best_val_loss:.6f}，喵呜~！(≧▽≦)✨")
