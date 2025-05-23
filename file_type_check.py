from charset_normalizer import from_path

result = from_path('corpus.txt')
best_guess = result.best()
print(f"编码: {best_guess.encoding}，置信度: {best_guess.fingerprint}")