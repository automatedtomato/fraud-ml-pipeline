# データ項目定義 (Data Description)

このドキュメントは、データ生成スクリプトによって生成され、PostgreSQLの`transactions`テーブルに格納されたデータ項目の定義と説明を記載します。

## テーブル: `transactions`

### 顧客情報 (Customer Information)

| カラム名 (Column Name) | データ型 (Data Type) | 説明 (Description) | 備考 (Notes) |
| :--- | :--- | :--- | :--- |
| `ssn` | text | 顧客の社会保障番号。 | 分析では直接使用しない可能性が高い。 |
| `cc_num` | bigint | クレジットカード番号。 | 顧客を識別するキーの一つ。 |
| `first` | text | 顧客の名。 | |
| `last` | text | 顧客の姓。 | |
| `gender` | text | 顧客の性別。（'M' または 'F'） | |
| `street` | text | 顧客の住所（通り）。 | |
| `city` | text | 顧客の住所（市）。 | |
| `state` | text | 顧客の住所（州）。 | |
| `zip` | bigint | 顧客の住所（郵便番号）。 | |
| `lat` | double precision | 顧客の居住地の緯度。 | |
| `long` | double precision | 顧客の居住地の経度。 | |
| `city_pop` | bigint | 居住都市の人口。 | |
| `job` | text | 顧客の職業。 | カテゴリカルな特徴量として利用可能。 |
| `dob` | text | 顧客の生年月日。 | DB上は`text`型。分析時に日付型に変換する。 |
| `acct_num` | bigint | 顧客の口座番号。 | |
| `profile` | text | データ生成時に使用された顧客プロファイル名。 | |

### 取引情報 (Transaction Information)

| カラム名 (Column Name) | データ型 (Data Type) | 説明 (Description) | 備考 (Notes) |
| :--- | :--- | :--- | :--- |
| `trans_num` | text | 取引ごとにユニークなID。 | |
| `trans_date` | text | 取引日 (YYYY-MM-DD)。 | DB上は`text`型。`trans_time`と結合して使用。|
| `trans_time` | text | 取引時刻 (HH:MM:SS)。 | DB上は`text`型。`trans_date`と結合して使用。|
| `unix_time` | bigint | UNIXタイムスタンプ形式の取引時刻。 | 時系列分析や特徴量エンジニアリングで非常に重要。 |
| `category` | text | 取引のカテゴリ。 | 例: `gas_transport`, `shopping_net`など。 |
| `amt` | double precision | 取引金額。 | |
| `is_fraud` | bigint | **不正利用フラグ（目的変数）** | **1が不正利用**、0が正常な取引。 |
| `merchant` | text | 取引が行われた加盟店名。 | |
| `merch_lat` | double precision | 加盟店の緯度。 | 顧客の居住地との距離計算などに利用可能。 |
| `merch_long` | double precision | 加盟店の経度。 | 顧客の居住地との距離計算などに利用可能。 |
