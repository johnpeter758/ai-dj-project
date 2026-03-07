# Reward Model for AI Music

## Key Signals
- Skip rate (most important)
- Completion rate
- Playlist adds
- Shares
- Replays

## Reward Mapping
- Skip <10s → -1
- Full play → +1
- Playlist add → +2
- Share → +3

## Skip Rate Optimization
```python
if listen_ratio < 0.1: return -2.0  # Instant skip
elif listen_ratio < 0.5: return -0.5  # Moderate
else: return listen_ratio  # Positive
```

## Engagement Metrics
- Danceability/energy
- Mood consistency
- Audio quality
- Novelty vs familiarity

## RLHF Pipeline
1. Generate N samples
2. Human rankings → preference data
3. Train reward model
4. Fine-tune with PPO

## Tools
- TRL (Hugging Face)
- AudioMAE
- WavLM
- Audiocraft
