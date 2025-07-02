# 不正利用検知MLパイプライン

クレジットカード不正利用検知に特化した、機械学習パイプラインの実装プロジェクトです。PostgreSQL + MLflow + FastAPI を用いたMLOpsワークフローの実践的な学習を目的としています。

## プロジェクト概要

"Sparkov"エンジンによって生成される擬似クレジットカード取引データを用いて、実践的な不正利用検知システムを構築します。

### 主な特徴
- **特徴量エンジニアリング**: 時系列やユーザーの行動パターンの分析
- **機械学習**: XGBoost, H2O AutoML, PyTorch転移学習の比較
- **評価**: Precision@Kのような実践的な指標を用いた性能評価
- **API**: FastAPIベースの不正スコアリングサービス
- **MLOps**: MLflowによる実験管理とモデルのバージョニング

## 技術スタック

- **バックエンド**: Python, FastAPI, SQLAlchemy, psycopg2
- **データベース**: PostgreSQL
- **機械学習**: PyTorch, XGBoost, H2O AutoML
- **MLOps**: MLflow, Docker, Docker Compose
- **開発・テスト**: Poetry, Jupyter, pytest

## セットアップと使い方

### 1. 前提条件
- Docker および Docker Compose
- Git

### 2. 初期セットアップ
1.  **リポジトリをクローン:**
    ```bash
    git clone [https://github.com/your-username/fraud-guardian-ml.git](https://github.com/your-username/fraud-guardian-ml.git)
    cd fraud-guardian-ml
    ```
2.  **`.env`ファイルの作成:**
    環境変数のサンプルファイルをコピーします。ローカル開発ではデフォルト値のままで問題ありません。
    ```bash
    cp .env.example .env
    ```
3.  **コンテナのビルドと起動:**
    開発用コンテナ、PostgreSQL、MLflowを起動します。
    ```bash
    docker-compose up -d --build
    ```

### 3. 開発環境
**VSCode Dev Containers**拡張機能の使用を強く推奨します。
1.  プロジェクトフォルダをVSCodeで開きます。
2.  プロンプトが表示されたら、「Reopen in Container」をクリックします。

### 4. 特徴量生成
生データを生成し、モデル学習用の特徴量テーブルを作成するには、ホストマシンのターミナルから以下のコマンドを実行します。
```bash
docker-compose exec dev poetry run python src/scripts/create_features.py
```

このコマンドは、特徴量エンジニアリングのパイプライン全体を実行します。生データを読み込み、全ての派生特徴量（Recency, Frequency, Monetaryなど）を計算し、最終的な`feature_transactions`テーブルを作成します。

### 5. データ確認

特徴量テーブルが作成されたことを確認するには、以下のコマンドを実行します。

```bash
docker-compose exec db psql -U user -d fraud_detection -c "SELECT COUNT(*) FROM feature_transactions;"
```

## 開発の進捗

  - ✅ **Sprint 1: インフラ基盤構築とEDA - 完了**
  - ✅ **Sprint 2: 特徴量エンジニアリングとパイプライン - 完了**
  - Sprint 3: 機械学習モデルの比較
  - Sprint 4: API開発と統合

## ライセンス

MIT License - 詳細はLICENSEファイルを参照してください。

## 作成者

Hikaru Tomizawa (富澤晃)

-----

*このプロジェクトは学習およびポートフォリオ目的で開発されています。*
