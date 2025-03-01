from model import CustomRWN

rwn = CustomRWN(device="cpu")
print(1)
rwn.train("slice_localization_data.csv")
