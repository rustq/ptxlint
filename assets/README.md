`promo.png` is rendered through [carbon.now.sh](https://carbon.now.sh) from `promo.txt`, which is lines taken from real ptxlint output.

```shell
cd assets && npm i playwright-core && CHROME=/path/to/chrome node shot.mjs promo.png
```
