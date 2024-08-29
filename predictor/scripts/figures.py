import matplotlib.pyplot as plt
import json

# Load the saved training history
with open('training_history.json', 'r') as f:
    history = json.load(f)

# Load performance metrics
metrics = {
    'Mean Absolute Error (MAE)': 196289.9809209623,
    'Mean Squared Error (MSE)': 750237527757691.6,
    'Root Mean Squared Error (RMSE)': 27390464.17565229,
    'R² Score': 0.9971849867584394
}

# Load cross-validation scores
cv_scores = [0.99475014, 0.99891294, 0.95971153, 0.93882353, 0.99414646]
mean_cv_score = 0.9772689180758519

plt.figure(figsize=(10, 6))
plt.plot(history['loss'], label='Training Loss')
# plt.plot(history['val_loss'], label='Validation Loss')  # Uncomment if val_loss is available
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('training_validation_loss.png')
plt.show()
# 2. Performance Metrics Plot
plt.figure(figsize=(14, 7))

plt.subplot(2, 2, 1)
plt.bar(['MAE'], [metrics['Mean Absolute Error (MAE)']], color='skyblue')
plt.title('Mean Absolute Error (MAE)')
plt.ylabel('Value')

plt.subplot(2, 2, 2)
plt.bar(['MSE'], [metrics['Mean Squared Error (MSE)']], color='skyblue')
plt.title('Mean Squared Error (MSE)')
plt.ylabel('Value')

plt.subplot(2, 2, 3)
plt.bar(['RMSE'], [metrics['Root Mean Squared Error (RMSE)']], color='skyblue')
plt.title('Root Mean Squared Error (RMSE)')
plt.ylabel('Value')

plt.subplot(2, 2, 4)
plt.bar(['R² Score'], [metrics['R² Score']], color='skyblue')
plt.title('R² Score')
plt.ylabel('Value')

plt.tight_layout()
plt.savefig('performance_metrics.png')
plt.show()

# 3. Cross-Validation Scores Plot
plt.figure(figsize=(7, 7))
plt.bar(range(len(cv_scores)), cv_scores, color='skyblue')
plt.axhline(y=mean_cv_score, color='r', linestyle='--', label='Mean CV Score')
plt.xticks(range(len(cv_scores)), [f'Fold {i+1}' for i in range(len(cv_scores))])
plt.ylim(0.9, 1.0)
plt.title('Cross-Validation Scores')
plt.xlabel('Fold')
plt.ylabel('Score')
plt.legend()

plt.tight_layout()
plt.savefig('cross_validation_scores.png')
plt.show()
