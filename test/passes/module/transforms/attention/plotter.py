import matplotlib.pyplot as plt
import numpy as np

kv_heads = [1, 2, 3, 4, 6, 12]
total_cache = [2.021484375, 4.04296875, 6.064453125, 8.0859375, 10.107421875, 12.12890625]
avg_cache = [1.0107421875, 2.021484375, 3.0322265625, 4.04296875, 5.0537109375, 6.064453125]

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(kv_heads, total_cache, 'o-', color='blue', label='Total KV Cache (2 samples)')
ax.plot(kv_heads, avg_cache, 's-', color='red', label='Average KV Cache')
ax.set_xlabel('Number of KV Heads')
ax.set_ylabel('KV Cache Size per sample/MB')
ax.set_title('KV Cache Size vs. Number of KV Heads')
ax.grid(True, linestyle='--', alpha=0.7)
ax.legend()

plt.tight_layout()
plt.savefig('kv_cache_plot.png', dpi=300, bbox_inches='tight')
plt.show()