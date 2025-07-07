# 不正検知MLパイプライン

クレジットカードの不正利用検知に特化した、エンドツーエンドの機械学習パイプライン実装プロジェクトです。MLOpsの原則、保守性、そして高度なモデリング技術の実践と学習を目的としています。

## プロジェクト概要

「Sparkov」エンジンによって生成された、130万件を超える大規模な模擬クレジットカードトランザクションデータを使用し、実践的な不正検知システムを構築します。

このプロジェクトの中核は、生のデータの生成から高度な特徴量エンジニアリング、そして3つの根本的に異なるモデルの学習と評価、APIデプロイ準備まで、機械学習のライフサイクル全体をカバーする、柔軟なパイプラインです。

比較・評価するモデルは、それぞれ異なるアプローチから不正検知問題に取り組みます。
1.  **XGBoost**: 業界標準とも言える、強力な勾配ブースティングモデル。
2.  **Isolation Forest**: 教師なしの異常検知モデル。メモリ問題を解決し、コンセプトドリフトに対応するため、カスタムの「時間窓アンサンブル」として実装。
3.  **PyTorch (TabNet)**: アテンション機構を用いた、テーブルデータのためのモダンな深層学習モデル。

### 主な特徴
- **モジュール化された保守性の高いアーキテクチャ**: コードベースは`trainer`と`evaluator`に責務分割されており、テスト、保守、拡張が容易です。
- **3モデル比較**: 勾配ブースティング、教師なし異常検知、深層学習という異なるアプローチを並行して評価し、それぞれの長所と短所を明らかにします。
- **高度な特徴量エンジニアリング**: PostgreSQLのウィンドウ関数を駆使し、洗練された時系列特徴量やユーザー行動特徴量を生成します。
- **メモリ効率の良いパイプライン**: チャンク単位の処理（逐次学習）やカスタムアンサンブル手法により、メモリに収まらない大規模データセットを扱います。
- **ビジネス指向の評価**: 不正調査の費用対効果に直結する`Precision@K`を主要な評価指標として重視し、実用的な観点からモデルを評価します。

### 技術スタック
- **バックエンド**: Python, FastAPI, SQLAlchemy, psycopg2-binary
- **データベース**: PostgreSQL
- **機械学習**: PyTorch (TabNet), XGBoost, Scikit-learn
- **データ操作**: Pandas, NumPy
- **MLOps**: MLflow, Docker, Docker Compose
- **開発環境**: Poetry, Jupyter, Pytest

### 環境構築と実行手順

#### 1. 前提条件
- Docker と Docker Compose
- Git

#### 2. 初期セットアップ
1.  **リポジトリのクローン:**
    ```bash
    git clone https://github.com/automatedtomato/fraud-guardian-ml.git
    cd fraud-ml-pipeline
    ```
2.  **`.env` ファイルの作成:**
    環境設定ファイルのサンプルをコピーします。ローカル開発ではデフォルト値のままで問題ありません。
    ```bash
    cp .env.example .env
    ```
3.  **コンテナのビルドと起動:**
    開発コンテナ、PostgreSQL、MLflowが起動します。
    ```bash
    docker-compose up -d --build

    # または、GPUが使用可能環境の場合
    docker-compose -f docker-compose.gpu.yml up -d
    ```

#### 3. 生データの生成
以下のコマンドを実行し、初期データとなる生のトランザクションをデータベースに生成します。
```bash
docker-compose exec app poetry run python src/scripts/generate_and_load.py
```

#### 4. 特徴量の生成
このスクリプトは、特徴量エンジニアリングのパイプライン全体を実行します。生データを読み込み、全てのカスタム特徴量を計算し、最終的な学習用テーブル`feature_transactions`を作成します。
```bash
docker-compose exec app poetry run python src/scripts/create_features.py
```

#### 5. 学習・評価パイプラインの実行
`config/models.yml`で指定された全てのモデルの学習と評価を行うメインスクリプトです。
```bash
docker-compose exec app poetry run python src/scripts/run_pipeline.py
```

#### 6. データ生成の確認 (任意)
特徴量テーブルが正しく作成されたことを確認できます。
```bash
docker-compose exec db psql -U user -d fraud_db -c "SELECT COUNT(*) FROM feature_transactions;"
```

### 開発環境について
**VSCode Dev Containers**拡張機能を使用すると、シームレスな開発体験が得られるため、強く推奨します。
1.  VSCodeでプロジェクトフォルダを開きます。
2.  プロンプトが表示されたら、「Reopen in Container」をクリックします。

### 開発進捗
  - ✅ **Sprint 1: インフラ構築とEDA - 完了**
  - ✅ **Sprint 2: 特徴量エンジニアリングとパイプライン構築 - 完了**
  - ✅ **Sprint 3: 機械学習モデルの比較・評価 - 完了**
  -  Sprint 4: API開発と統合

### ライセンス
MIT License - 詳細はLICENSEファイルを参照してください。

### 著者
Hikaru Tomizawa (富澤晃)

-----

*このプロジェクトは、学習およびポートフォリオ目的で開発されています。*