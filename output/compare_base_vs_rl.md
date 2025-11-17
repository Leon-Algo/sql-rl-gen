# 基座模型 vs 强化学习后模型 对比样本表

| 样本序号 | gold SQL (expected) | 基座模型 SQL (output) | 基座 accuracy | RL SQL (output) | RL accuracy |
| -------- | ------------------- | --------------------- | ------------- | --------------- | ----------- |
| 1 | `SELECT count(*) FROM singer` | `SELECT count(*) FROM singer` | 1.0 | `SELECT count(*) FROM singer` | 1.0 |
| 2 | `SELECT count(*) FROM singer` | `SELECT count(*) FROM singer` | 1.0 | `SELECT count(*) FROM singer` | 1.0 |
| 3 | `SELECT name ,  country ,  age FROM singer ORDER BY age DESC` | `SELECT name, country, age FROM singer ORDER BY age DESC` | 1.0 | `SELECT name, country, age FROM singer ORDER BY age DESC` | 1.0 |
| 4 | `SELECT name ,  country ,  age FROM singer ORDER BY age DESC` | `SELECT name, country, age FROM singer ORDER BY age DESC` | 1.0 | `SELECT name, country, age FROM singer ORDER BY age DESC` | 1.0 |
| 5 | `SELECT avg(age) ,  min(age) ,  max(age) FROM singer WHERE country  =  'France'` | `SELECT avg(age), min(age), max(age) FROM singer WHERE country = "France"` | 1.0 | `SELECT avg(age), min(age), max(age) FROM singer WHERE country = "France"` | 1.0 |
| 6 | `SELECT avg(age) ,  min(age) ,  max(age) FROM singer WHERE country  =  'France'` | `SELECT avg(age), min(age), max(age) FROM singer WHERE country = "France"` | 1.0 | `SELECT avg(age), min(age), max(age) FROM singer WHERE country = "France"` | 1.0 |
| 7 | `SELECT song_name ,  song_release_year FROM singer ORDER BY age LIMIT 1` | `SELECT song_name, song_release_year FROM singer ORDER BY age LIMIT 1` | 1.0 | `SELECT song_name, song_release_year FROM singer ORDER BY age LIMIT 1` | 1.0 |
| 8 | `SELECT song_name ,  song_release_year FROM singer ORDER BY age LIMIT 1` | `SELECT song_name, song_release_year FROM singer ORDER BY age LIMIT 1` | 1.0 | `SELECT song_name, song_release_year FROM singer ORDER BY age LIMIT 1` | 1.0 |
| 9 | `SELECT DISTINCT country FROM singer WHERE age  >  20` | `SELECT DISTINCT country FROM singer WHERE age > 20` | 1.0 | `SELECT DISTINCT country FROM singer WHERE age > 20` | 1.0 |
| 10 | `SELECT DISTINCT country FROM singer WHERE age  >  20` | `SELECT DISTINCT country FROM singer WHERE age > 20` | 1.0 | `SELECT DISTINCT country FROM singer WHERE age > 20` | 1.0 |
| 11 | `SELECT country ,  count(*) FROM singer GROUP BY country` | `SELECT country, count(*) FROM singer GROUP BY country` | 1.0 | `SELECT country, count(*) FROM singer GROUP BY country` | 1.0 |
| 12 | `SELECT country ,  count(*) FROM singer GROUP BY country` | `SELECT country, count(*) FROM singer GROUP BY country` | 1.0 | `SELECT country, count(*) FROM singer GROUP BY country` | 1.0 |
| 13 | `SELECT song_name FROM singer WHERE age  >  (SELECT avg(age) FROM singer)` | `SELECT song_name FROM singer WHERE age > (SELECT avg(age) FROM singer)` | 1.0 | `SELECT song_name FROM singer WHERE age > (SELECT avg(age) FROM singer)` | 1.0 |
| 14 | `SELECT song_name FROM singer WHERE age  >  (SELECT avg(age) FROM singer)` | `SELECT song_name FROM singer WHERE age > (SELECT avg(age) FROM singer)` | 1.0 | `SELECT song_name FROM singer WHERE age > (SELECT avg(age) FROM singer)` | 1.0 |
| 15 | `SELECT LOCATION ,  name FROM stadium WHERE capacity BETWEEN 5000 AND 10000` | `SELECT LOCATION, name FROM stadium WHERE capacity BETWEEN 5000 AND 10000` | 1.0 | `SELECT LOCATION, name FROM stadium WHERE capacity BETWEEN 5000 AND 10000` | 1.0 |
| 16 | `SELECT LOCATION ,  name FROM stadium WHERE capacity BETWEEN 5000 AND 10000` | `SELECT LOCATION, name FROM stadium WHERE capacity BETWEEN 5000 AND 10000` | 1.0 | `SELECT LOCATION, name FROM stadium WHERE capacity BETWEEN 5000 AND 10000` | 1.0 |
| 17 | `select max(capacity), average from stadium` | `SELECT max(capacity), avg(capacity) FROM stadium` | 0.0 | `SELECT max(capacity), avg(capacity) FROM stadium` | 0.0 |
| 18 | `select avg(capacity) ,  max(capacity) from stadium` | `SELECT avg(capacity), max(capacity) FROM stadium` | 1.0 | `SELECT avg(capacity), max(capacity) FROM stadium` | 1.0 |
| 19 | `SELECT name ,  capacity FROM stadium ORDER BY average DESC LIMIT 1` | `SELECT name, capacity FROM stadium ORDER BY avg(attendance) DESC LIMIT 1` | 0.0 | `SELECT name, capacity FROM stadium ORDER BY avg(attendance) DESC LIMIT 1` | 0.0 |
| 20 | `SELECT name ,  capacity FROM stadium ORDER BY average DESC LIMIT 1` | `SELECT name, capacity FROM stadium ORDER BY avg(attendance) DESC LIMIT 1` | 0.0 | `SELECT name, capacity FROM stadium ORDER BY avg(attendance) DESC LIMIT 1` | 0.0 |
