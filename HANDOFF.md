# Devir notu: Crypto Forecasting Live

Son güncelleme: 2026-11-24 18:00 UTC. Bu dosya, önceki oturum kapatılırken yeni bir ajanın işe nereden devam edeceğini bilmesi için yazıldı. Önce bu dosyayı, sonra `README.md` ve `reports/backtest.md` dosyalarını oku.

## Proje nedir
BTC-USD'nin bir sonraki 4 saatlik mumunun yönünü tahmin eden bir XGBClassifier. Her tahmin, mum kapanmadan **önce** git'e commit edilir. Böylece git geçmişi sonradan değiştirilemeyen, zaman damgalı bir kayıt olur. Tahminlerle long/flat bir paper portföy işletilir: sanal $10.000, işlem başına %0,10 komisyon ve %0,05 kayma. Sonuç al-tut ile kıyaslanır.

Sahibinin kişisel sitesinde canlı gösterilir: https://suleymankara.com/projects/crypto-forecasting-live/ (TR: `/tr/projects/...`). Site deposu `suleyman-kara/personal-website`.

Proje, sahibinin staj projesi `suleyman-kara/retail-sales-forecasting`'in devamıdır. Aynı yaklaşım kullanılır: XGBoost, lag ve takvim özellikleri, MRMR/RFE/SFS, BayesSearchCV. O projenin README'sindeki "Known Limitations" maddeleri burada düzeltildi: walk-forward doğrulama, dokunulmayan holdout ve `TimeSeriesSplit(gap=1)`.

## Durum: tamamlandı ve yayında
| Parça | Yer | Durum |
|---|---|---|
| Veri, özellikler, walk-forward backtest, testler, CI | bu repo, PR #1 (birleşti) | ✅ |
| Holdout (son 6 ay), **bir kez** değerlendirildi | `reports/backtest.md`, `reports/backtest.json` → `holdout` | ✅ Yeniden açılmamalı ve tasarım bu sonuca göre değiştirilmemeli |
| Final model (holdout dahil tüm veri) | `models/model.json`, `models/meta.json` (sürüm `2026-11-23-2000`) | ✅ |
| Canlı tahmin, paper portföy, `status.json` | bu repo, PR #2 (birleşti) | ✅ ama cron sorunu var (aşağıda) |
| Site kartı (`LiveForecast.tsx`), proje kaydı, staj projesiyle bağlantı | personal-website, PR #4 (birleşti) | ✅ |

### Sonuçlar (dürüstçe yayınlandı, ayar yapılmadı)
| | İsabet | AUC | Masraf sonrası getiri | Max DD |
|---|---:|---:|---:|---:|
| Walk-forward modeli (2017-08 → 2026-05) | %53,66 | 0,554 | −%93,6 | −%97,8 |
| Aynı dönemde al-tut | – | – | +%2.730 | −%84,0 |
| Holdout modeli (2026-05 → 2026-11) | %53,08 | 0,547 | −%6,3 | −%7,2 |
| Holdout döneminde al-tut | – | – | −%19,1 | −%34,7 |

Bulgu: Sınıflandırıcının yön tahmininde küçük ama istatistiksel olarak anlamlı bir üstünlüğü var. 4 saatlik long/flat stratejide işlem masrafları bu üstünlüğü yok ediyor. Sahibi "sonuç zayıfsa bile dürüstçe yayınla" kararını verdi. **Backtest sonuçlarını güzelleştirmek için yeniden ayar yapma.**

## ⚠️ Açık sorun (sıradaki iş): GitHub cron'u güvenilmez
`.github/workflows/predict.yml` cron'u `7 */4 * * *`. 24 Kasım'da 04:07, 08:07 ve 12:07 çalışmaları **hiç başlamadı**. Tek zamanlanmış çalışma 14:45'te, yani yaklaşık 2,5 saat geç geldi. `live.py`'deki dürüstlük kuralı gereği, hedef mum açıldıktan 1 saatten fazla sonra başlayan bir çalışma tahmin yapmaz, slotu `skipped` yazar. Bu yüzden 04:00, 08:00 ve 12:00 slotları atlandı. Bu sorun çözülmezse kayıt çoğunlukla "skipped" satırlarından oluşur.

Önerilen çözümler (birini seç ve sahibine söyle):
1. **En basit çözüm:** cron'u sıklaştır, örneğin `7,22,37,52 * * * *`. Script idempotent olduğu için yeni bir şey yoksa commit atmadan çıkar. Halka açık repoda Actions dakikaları ücretsiz. Böylece her 4 saatlik pencerenin ilk saatinde en az bir çalışmanın denk gelme olasılığı çok artar.
2. **En güvenilir çözüm:** Dış bir zamanlayıcı her 4 saatte `workflow_dispatch` veya `repository_dispatch` tetikler. Örneğin bir Cloudflare Worker Cron Trigger. Bunun için ince yetkili bir GitHub token'ı gerekir ve token Worker'ın gizli değişkeni olarak saklanır. Sahibinin onayı ve kurulumu gerekir.
3. İkisinin birleşimi.

**1 saatlik kuralı gevşetme.** Kural, sonucu kısmen görünür olan mumu tahmin etmeyi engelliyor. Projenin bütün iddiası bu dürüstlüğe dayanıyor.

## Diğer takip edilecekler
- `retrain.yml` her Pazar 02:23 UTC'de çalışır. İlk çalışmasının başarılı olduğunu kontrol et.
- GitHub, 60 gün aktivite olmayan repolarda zamanlanmış workflow'ları kapatır. Botun commit'leri repoyu aktif tutmalı, yine de ara ara kontrol et.
- Site, `status.json`'u tarayıcıda `https://raw.githubusercontent.com/suleyman-kara/crypto-forecasting-live/main/live/status.json` adresinden canlı çekiyor (yaklaşık 5 dakika önbellek). Build sırasındaki yedek kopya `personal-website/src/data/forecast.json`, deploy'da `scripts/fetch-forecast.mjs` ile yenileniyor. Günlük deploy cron'u var.
- Kart, 30 tahmin sonuçlanana kadar "veri toplanıyor" etiketi gösterir. Getiri grafiği ancak 2 nokta olunca çizilir.

## `live/status.json` sözleşmesi (schema 1). Site tam olarak bunu okuyor, alan değiştirirsen siteyi de güncelle
- `accuracy` ve `accuracy30d` 0–1 arası oran. `*Pct` alanları yüzde biriminde (12.3 = %12,3), düşüş negatif.
- `current` hedef mumu anlatır: `candleOpen` / `targetClose` tahmin edilen mumun açılış ve kapanışıdır. `p` = P(yükseliş). `threshold`, long'a geçilen p seviyesidir, yani **0.5 + m** (şu an 0,6). `meta.json`'daki `threshold` alanı ise m'dir (0,1).
- `model` henüz model yoksa `null` olabilir.
- `recent`: en yeni başta, en fazla 20 kayıt. `skipped` satırlarında `p`, `direction` ve `action` alanları `null`. `realizedReturnPct`, hedef mumun kendi getirisidir (portföyün değil). Sonuç beklenirken veya atlanmışken `null`.
- "hit": tahmin edilen yön, hedef mumun getirisinin işaretiyle uyuştu demek.
- `backtest` bloğu `reports/backtest.json` içindeki walk-forward rakamlarından gelir.

## Tasarım kararları ve plandan ayrılan yerler
- Veri kaynağı Coinbase Exchange'in açık API'si. Binance, GitHub'ın ABD'deki sunucularını engelliyor. 1 saatlik mumlar 4 saatliğe çevriliyor ve yalnızca kapanmış mumlar kullanılıyor.
- Yeni mumlar, yalnızca sonuna eklenen `data/candles_1h_live.csv` dosyasına yazılıyor. 4 MB'lık parquet artık yeniden yazılmıyor, aksi halde git geçmişi şişer. `load_1h` iki dosyayı da okur.
- Kaçırılan slotlar geriye dönük doldurulmaz, `skipped` olarak yazılır. Atlanan slotta önceki paper pozisyon korunur.
- `retrain.yml` yeni mum çekmez, commit edilmiş mumlarla eğitir. Böylece iki workflow arasında push çakışması olmaz.
- Yeniden eğitilen model, tahmin kayıtlarında `model_version` ile izlenir.

## Yerelde çalıştırma
```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt && pip install -e .
pytest -q                       # 36 test, ağ gerektirmez
python scripts/backtest.py      # walk-forward (~9 dk); holdout bayrağı kapalı kalmalı
python scripts/train.py         # final model -> models/
python scripts/predict.py       # canlı adım (Coinbase'e erişim gerekir); idempotent
```

## Çalışma kuralları (sahibinin tercihleri)
- Sahibiyle **Türkçe** iletişim kur.
- Uygun işlerde alt ajan kullan (opus, effort medium) ve ana bağlamı temiz tut.
- Değişiklikleri ayrı bir dalda yap ve PR aç. Sahibi önceki oturumda testler geçince `main`'e birleştirme izni vermişti. Yeni oturumda kapsamını teyit et.
- Commit mesajlarına oturumun verdiği attribution satırlarını ekle.
