from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import torch
from model import *

best_params = load_best_hyperparams()


def evaluate_model(model, dataloader, criterion):
    """评估模型，计算 R²、MSE 和 RMSE"""
    model.eval()  # 进入评估模式
    all_preds, all_targets = [], []
    total_loss = 0

    with torch.no_grad():
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)

            loss = criterion(outputs, batch_y)
            total_loss += loss.item()

            all_preds.extend(outputs.cpu().numpy().flatten())
            all_targets.extend(batch_y.cpu().numpy().flatten())

    mse = mean_squared_error(all_targets, all_preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(all_targets, all_preds)

    print("📊✨ 模型评估完毕喵！来看看结果吧！(ฅ^•ω•^ฅ)")
    print(f"🔹 MSE（均方误差）: {mse:.6f}")
    print(f"🔹 RMSE（均方根误差）: {rmse:.6f}")
    print(f"🔹 R²（决定系数）: {r2:.6f} (R² 越接近 1 说明模型拟合效果越好哦！)")

    return {"MSE": mse, "RMSE": rmse, "R²": r2}

# 加载最佳模型
final_model = RegressionModel(input_size=X.shape[1], hidden_size=best_params["hidden_size"], num_layers=best_params["num_layers"]).to(device)
final_model.load_state_dict(torch.load("best_model.pth"))
final_model.eval()

# 评估模型
test_loader = data.DataLoader(val_dataset, batch_size=best_params["batch_size"], shuffle=False)
metrics = evaluate_model(final_model, test_loader, nn.MSELoss())
