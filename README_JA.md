# Fraud Guardian ML

詐欺検知に特化した機械学習パイプラインの実装プロジェクト。PostgreSQL + MLflow + FastAPI を使用したMLOpsワークフローの学習・実践を目的としています。

## 🎯 プロジェクト概要

130万件のクレジットカード取引データ（sparknov dataset）を使用し、実用的な詐欺検知システムを構築します。

### 主要機能
- **特徴量エンジニアリング**: 時系列・ユーザー行動パターンの分析
- **機械学習**: XGBoost、H2O AutoML、PyTorch転移学習の比較
- **評価**: Precision@K による実務指標での性能評価
- **API**: FastAPI による詐欺スコア算出サービス
- **MLOps**: MLflow による実験管理・モデルバージョニング

## 技術スタック

- **Backend**: Python, FastAPI, SQLAlchemy, psycopg2
- **Database**: PostgreSQL
- **ML**: PyTorch, XGBoost, H2O AutoML
- **MLOps**: MLflow, Docker-compose
- **Testing**: pytest, coverage

## クイックスタート

```bash
# リポジトリクローン
git clone https://github.com/your-username/fraud-guardian-ml.git
cd fraud-guardian-ml

# Docker環境起動
docker-compose up -d
```

## 開発進捗

- Sprint 1: インフラ基盤構築・EDA
- Sprint 2: 特徴量エンジニアリング
- Sprint 3: 機械学習モデル比較
- Sprint 4: API開発・統合

## ライセンス
MIT License - 詳細は LICENSE ファイルを参照

## 作者
富澤晃 (Hikaru Tomizawa)
