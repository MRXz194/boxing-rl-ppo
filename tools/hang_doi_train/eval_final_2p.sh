#!/bin/bash
# Đánh giá đầy đủ model 2 người cuối cùng (league6 = v6). Chạy: bash eval_final_2p.sh
cd "$(dirname "$0")/../.."
P=./.venv/Scripts/python.exe
M=models/league6/final.pt
F="Loading roms|A.L.E|Stella"
$P evaluate_2p.py vs-random --model $M --games 30 2>&1 | grep -vE "$F" | tail -2
$P evaluate_2p.py vs-cpu --model $M --episodes 30 2>&1 | grep -vE "$F" | tail -2
$P evaluate_2p.py match --a $M --b $M --games 10 --out results/2p_selfplay_league6.json 2>&1 | grep -vE "$F" | tail -2
$P evaluate_2p.py match --a $M --b models/league/final.pt --games 20 2>&1 | grep -vE "$F" | tail -2
$P evaluate_2p.py match --a $M --b models/ppo10M_s0/final.pt --games 20 2>&1 | grep -vE "$F" | tail -2
$P evaluate_2p.py match --a $M --b models/sp_pool/final.pt --games 20 2>&1 | grep -vE "$F" | tail -2
S1=$(ls models/league6/snapshots/snap_0010*.pt | head -1); S2=$(ls models/league6/snapshots/snap_0020*.pt | head -1)
$P evaluate_2p.py league --games 8 --name final2 --players random models/ppo10M_s0/final.pt models/sp_pool/final.pt \
   models/league/final.pt $S1 $S2 $M 2>&1 | grep -vE "$F" | tail -10
$P play_2p.py --white $M --black $M --out videos/2p_league6_vs_league6.mp4 2>&1 | tail -1
$P play_2p.py --white models/ppo10M_s0/final.pt --black $M --out videos/2p_ppo1P_vs_league6.mp4 2>&1 | tail -1
$P play_2p.py --white $M --black random --out videos/2p_league6_vs_random.mp4 2>&1 | tail -1
echo "XONG ĐÁNH GIÁ"
