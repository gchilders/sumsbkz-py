# Data archive: new records for equal sums of like powers

This archive accompanies the paper *New records for equal sums of like powers via lattice sieving* (J. G. Childers). It certifies the improved entries of J.-C. Meyrignac's table of minimal equal sums of like powers (http://euler.free.fr). The certificates are verifiable without any of the search software.

## Files

| file | contents |
|---|---|
| `identities.txt` | One identity per improved entry (m ≤ 16, 17 ≤ k ≤ 100), in the format of Meyrignac's database: `(k,m,n) a1+a2*2+...=b1+...`, where `b*c` stands for c copies of b^k. |
| `improvements.csv` | For each identity: k, m, n; the reference value it improves; whether that table entry was attained by an identity or was a derived bound; the saving; the largest base; and the signed relation vector as `base:coefficient` pairs (Σ c_x x^k = 0). |
| `table_reference.csv` | The reference data for all entries with m ≤ 16 and 17 ≤ k ≤ 100 in the two tables: the table value, its kind, and the least n of an identity of that type in Meyrignac's database. It was extracted from the site as accessed on 27 October 2026; the database was last modified in November 2022. |
| `totals.txt` | Identities whose total number of terms m+n is below that of every identity for the same k in Meyrignac's database and the 2003 archive (any m, one per k). |
| `totals_reference.csv` | For each k, the least m+n of an identity in Meyrignac's database and the 2003 archive. |
| `verify.py` | Standalone checker (Python 3, standard library only). |
| `SHA256SUMS` | Checksums of the files above, and of the snapshot of Meyrignac's pages and database files (`snapshot/...`) from which `table_reference.csv` was extracted. |

## Verification

```
python3 verify.py identities.txt table_reference.csv
```

For every identity the program checks:

- in exact integer arithmetic, that both sides are equal;
- that no base occurs on both sides;
- that the numbers of terms are (m, n) with m ≤ n;
- that the bases have no common factor (the identity is primitive);
- that n is smaller than the reference value, which is the smaller of the table entry and the least n of a database identity of the same type.

The identities in `totals.txt` receive the same checks, and their total m+n is compared with `totals_reference.csv`.

It prints the number of improved entries and the terms saved, separately for entries attained by an identity and for derived bounds. It exits with status 0 only if every check passes.

The reference value is the table entry, or the best identity of that type in the database when that is smaller. A *derived bound* is a table entry obtained from a neighboring entry by the rule that an identity of type (k,m,n) gives the bound n+c for (k,m+c); such an entry has no identity of its own type at that value.
