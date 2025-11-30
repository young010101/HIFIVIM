```python
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Now your original code works
from dipy.data import get_fnames
fraw, fbval, fbvec = get_fnames('ivim')   # ← will succeed now
```