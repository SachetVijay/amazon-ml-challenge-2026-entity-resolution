import hashlib
import os
import zipfile

D = os.environ.get("WIN_SUBMIT_DIR", "/mnt/c/Users/YOURUSER/Downloads/")
raw = open(D + "matching_results.tsv", "rb").read()
print(f"Downloads/matching_results.tsv: {len(raw)} bytes, {raw.count(b'\n')} lines "
      f"(full file = 1732545 lines), ends with newline: {raw.endswith(b'\n')}")
ours = {"test_source1.tsv": "51a155b8c84bfefcf049e983f06c3cf4", "test_source2.tsv": "ff8aba5701829a622ac05b2765c4eb5b",
        "test_source3.tsv": "5f48d48594dea3ea1455549eb5233ad8", "train_source1.tsv": "1a0c98e43dad1babed3c99bca2c7b5bd",
        "train_source2.tsv": "a7bddba33ead4a85e6c088ad3b377fc2", "train_source3.tsv": "5df07bf109f55a292bdfaa8c983cfa20",
        "train_ground_truth.tsv": "3643443570b2c881c425d69c2d46e95d"}
z = zipfile.ZipFile(D + "6ab10eb3b23ba_student_resource.zip")
for i in z.infolist():
    name = i.filename.split("/")[-1]
    if i.is_dir():
        continue
    if name in ours:
        h = hashlib.md5()
        with z.open(i) as f:
            for chunk in iter(lambda: f.read(1 << 22), b""):
                h.update(chunk)
        print(f"{i.filename:55s} {i.file_size:>11d}  {'SAME' if h.hexdigest() == ours[name] else 'DIFFERENT ' + h.hexdigest()}")
    else:
        print(f"{i.filename:55s} {i.file_size:>11d}  (other file)")
