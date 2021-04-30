import torch

is_probability_tensor = True

n_array_a = torch.tensor([0.8, 0.9, 0.8], dtype=torch.float)
n_array_b = torch.tensor([0.9, 0.9, 0.9], dtype=torch.float)

if is_probability_tensor:
    probability_a = n_array_a
    probability_b = n_array_b
else:
    probability_a = n_array_a.softmax(dim=0)
    probability_b = n_array_b.softmax(dim=0)

entropy_a = -probability_a * torch.log10(probability_a)
entropy_b = -probability_b * torch.log10(probability_b)

scholar_a = entropy_a.sum().cpu().numpy()
scholar_b = entropy_b.sum().cpu().numpy()

print('\nentropy_a :', scholar_a)
print('entropy_b :', scholar_b)

print('entropy_a' if scholar_a > scholar_b else 'entropy_b', 'is big')

print('\n---------Conclusion---------\n'
      'Data that tilted to specific dimension has low entropy.\n'
      'Maximizing entropy will make data flatten to each dimension.')
