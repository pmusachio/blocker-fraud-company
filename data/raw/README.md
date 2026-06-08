# Dados

Fonte Kaggle: [Synthetic Financial Datasets For Fraud Detection / PaySim](https://www.kaggle.com/datasets/ntnu-testimon/paysim1).

Arquivos esperados nesta pasta:

- `transactions.csv`

- O CSV do PaySim deve ser salvo como `data/raw/transactions.csv`.

## Download via Kaggle API

```bash
mkdir -p data/raw
kaggle datasets download -d ntnu-testimon/paysim1 --unzip -p data/raw
find data/raw -maxdepth 1 -name "*.zip" -exec unzip -q -o {} -d data/raw \;
```

Ajuste de nomes esperado pelo projeto:

```bash
mv data/raw/PS_20174392719_1491204439457_log.csv data/raw/transactions.csv 2>/dev/null || true
```

Mantenha arquivos grandes fora do Git quando necessario e baixe-os novamente no Colab ou no ambiente local.
