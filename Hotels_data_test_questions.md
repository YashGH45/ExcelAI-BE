# Hotels Data — Text-to-SQL Chatbot Test Questions

**Dataset:** `Hotels_data.xlsx`  
**Rows:** 134 &nbsp;|&nbsp; **Columns:** 7 (`person_name`, `company`, `room_number`, `check_in`, `check_out`, `duration`, `total_bill`)  
**Date range:** Jan 2022 – Dec 2022  
**Total bill range:** 2,005 – 9,984

---

## Category 1 — Simple Counts & Totals

### Q1. How many hotel bookings are there in total?
**Expected:** 134 bookings

---

### Q2. What is the total revenue from all bookings?
**Expected:** 792,042

---

### Q3. How many guests stayed for exactly 1 night?
**Expected:** 27 guests

---

### Q4. How many bookings were made in Q4 (October–December)?
**Expected:** 38 bookings

---

### Q5. How many check-outs happened in January 2022?
**Expected:** 12 check-outs

---

## Category 2 — Aggregations (AVG, MIN, MAX)

### Q6. What is the average total bill across all bookings?
**Expected:** 5,910.76

---

### Q7. What is the average length of stay?
**Expected:** 2.60 nights

---

### Q8. Who had the highest total bill, and how much was it?
**Expected:** Lucie Jewiss — Room 207 — 9,984 (Company: Brightdog)

---

### Q9. What is the cheapest bill recorded? Who was the guest?
**Expected:** 2,005 — Ladonna Castagna (Room 100, 3 nights)

---

### Q10. What is the average bill for each stay duration (1, 2, 3, 4 nights)?
**Expected:**

| Duration | Avg Total Bill |
|----------|---------------|
| 1 night  | 5,831.15      |
| 2 nights | 5,848.91      |
| 3 nights | 6,377.98      |
| 4 nights | 5,484.35      |

---

## Category 3 — Filtering

### Q11. List all guests who checked in after December 1, 2022.
**Expected:** 13 guests, including:
- Deina Harwin (Dec 13, Room 300, 9,468)
- Benyamin Crocetti (Dec 26, Room 210, 4,968)
- Aubert Racher (Dec 17, Room 209, 8,503)
- Rodrique Brockbank (Dec 12, Room 405, 8,420)
- *(and 9 others)*

---

### Q12. Show all bookings for Room 304.
**Expected:** 3 bookings

| Guest           | Check In   | Check Out  | Duration | Total Bill |
|-----------------|------------|------------|----------|------------|
| Stanley Hadrill | 2022-05-20 | 2022-05-22 | 2        | 2,785      |
| Lyndell Tice    | 2022-02-23 | 2022-02-25 | 2        | 2,028      |
| Nehemiah Huke   | 2022-07-29 | 2022-08-02 | 4        | 4,849      |

---

### Q13. List guests who stayed 4 nights and had a total bill over 8,000.
**Expected:** 5 guests

| Guest              | Room | Total Bill |
|--------------------|------|------------|
| Aubert Racher      | 209  | 8,503      |
| Angeline Goodbairn | 303  | 9,787      |
| Chickie Altimas    | 204  | 9,455      |
| Junie Parrott      | 204  | 9,556      |
| Pierce Meese       | 105  | 8,721      |

---

### Q14. How many guests have a name starting with the letter "A"?
**Expected:** 17 guests

---

## Category 4 — Grouping & Ranking

### Q15. How many guests stayed on each floor? (Floor 1 = rooms 100–109, Floor 2 = 200–209, etc.)
**Expected:**

| Floor | Guests |
|-------|--------|
| 1st   | 41     |
| 2nd   | 31     |
| 3rd   | 27     |
| 4th   | 35     |

---

### Q16. Which company spent the most in total across all their bookings?
**Expected:**

| Company   | Total Spend |
|-----------|-------------|
| Rhynyx    | 19,157      |
| Brightdog | 18,902      |
| Skinix    | 18,292      |

---

### Q17. What is the total bill for all bookings made by Leexo?
**Expected:** 14,943 (3 stays)

---

### Q18. Which month had the most check-ins?
**Expected:** September — 14 check-ins  
*(January and November are tied second with 13 each)*

---

### Q19. Which room number has the highest average total bill?
**Expected:**

| Room | Avg Bill |
|------|----------|
| 400  | 9,357.00 |
| 301  | 8,954.00 |
| 201  | 8,028.00 |
| 409  | 7,877.40 |
| 308  | 7,831.50 |

---

### Q20. What is the highest total bill on each floor?
**Expected:**

| Floor | Max Bill | Guest              |
|-------|----------|--------------------|
| 1st   | 9,890    | Gertie Dominguez   |
| 2nd   | 9,984    | Lucie Jewiss       |
| 3rd   | 9,804    | Sandor Sagg        |
| 4th   | 9,830    | Heda Burcombe      |

---

## Category 5 — Date & Time Queries

### Q21. How many bookings were made in each month of 2022?
**Expected:**

| Month     | Bookings |
|-----------|----------|
| January   | 13       |
| February  | 12       |
| March     | 7        |
| April     | 12       |
| May       | 12       |
| June      | 9        |
| July      | 7        |
| August    | 10       |
| September | 14       |
| October   | 11       |
| November  | 14       |
| December  | 13       |

---

### Q22. Compare bookings in September vs January. Which had more?
**Expected:** Both tied at **14 and 13** respectively — September had 1 more booking than January.

---

### Q23. Which guests checked in during the first week of 2022 (Jan 1–7)?
**Expected:** Guests with `check_in` between 2022-01-03 and 2022-01-07  
*(Check against check_in column — min date in dataset is 2022-01-03)*

---

## Category 6 — Multi-condition & Complex Queries

### Q24. Which rooms were booked more than 5 times?
**Expected:**

| Room | Bookings |
|------|----------|
| 103  | 8        |
| 109  | 7        |
| 209  | 7        |
| 204  | 6        |
| 307  | 6        |
| 410  | 6        |

---

### Q25. Who are the top 5 guests by total bill amount?
**Expected:**

| Rank | Guest              | Company   | Room | Total Bill |
|------|--------------------|-----------|------|------------|
| 1    | Lucie Jewiss       | Brightdog | 207  | 9,984      |
| 2    | Gertie Dominguez   | Dabshots  | 103  | 9,890      |
| 3    | Heda Burcombe      | Centidel  | 409  | 9,830      |
| 4    | Sandor Sagg        | Voolia    | 306  | 9,804      |
| 5    | Angeline Goodbairn | Ntags     | 303  | 9,787      |

---

### Q26. What is the total revenue from 3rd-floor rooms (300–309)?
**Expected:** Sum of `total_bill` where `room_number` between 300 and 309

---

### Q27. List all guests from companies with more than 2 bookings.
**Expected:** Guests from Leexo, Quinu, or Fivechat (each with 3 bookings)

---

## Edge Cases & Potential Failure Points

| # | Question | What it tests |
|---|----------|---------------|
| E1 | "Show me bookings for room 999" | Zero-row result / empty response |
| E2 | "Which guests stayed more than 4 nights?" | No rows — max duration is 4 |
| E3 | "What is the total bill for company XYZ?" | Non-existent company |
| E4 | "List guests sorted by check-in date descending" | ORDER BY on datetime |
| E5 | "What is the bill per night for each booking?" | Derived column (`total_bill / duration`) |
| E6 | "How many unique companies are there?" | COUNT DISTINCT |
| E7 | "Show the first 10 bookings ordered by total bill" | LIMIT + ORDER BY |
| E8 | "Which guests checked in and out in the same month?" | Date function on two columns |
