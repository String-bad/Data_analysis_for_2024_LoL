# Import necessary libraries
import pandas as pd

# Load the dataset
file_path = 'player_statistics_cleaned_final.csv'
data = pd.read_csv(file_path)

# Check the current state of 'Solo Kills' column
print("Before cleaning:\n", data['Solo Kills'].value_counts())

# Clean 'Solo Kills' column
# Convert non-numeric entries to NaN and replace them with 0
data['Solo Kills'] = pd.to_numeric(data['Solo Kills'], errors='coerce').fillna(0).astype(int)

# Verify the cleaning process
print("After cleaning:\n", data['Solo Kills'].value_counts())

# Save the cleaned dataset (optional)
cleaned_file_path = 'player_statistics_cleaned_solo_kill.csv'
data.to_csv(cleaned_file_path, index=False)

print(f"Cleaned data saved to: {cleaned_file_path}")

# 数据集整体情况
# TeamName:队名
# PlayerName:选手名
# Position:位置
# Games:比赛场次
# Win Rate:胜率
# KDA:击杀-死亡-助攻比，衡量选手表现
# Avg kills:平均击杀
# Avg deaths:平均死亡
# Avg assists:平均助攻
# CSPerMin:每分钟补兵数
# GoldPerMin:每分钟金钱数
# KP%:参团率
# DamagePercent:伤害占比
# DPM:每分钟对英雄伤害输出
# VSPM:每分钟视野得分
# Avg WPM:每分钟放置眼位（真眼+假眼）平均数量
# Avg WCPM:每分钟放置眼位（真眼）平均数量
# Avg VWPM:每分钟排除眼位平均数量
# GD@15:15分钟经济领先/落后
# CSD@15:15分钟补兵领先/落后
# XPD@15:15分钟经验领先/落后
# FB %:每场比赛一血率
# FB Victim:每场比赛被一血率
# Penta Kills:五杀次数
# Solo Kills:单杀次数
# Country:国家
# FlashKeybind:闪现按键
