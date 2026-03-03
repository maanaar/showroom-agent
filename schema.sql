-- Ibyco Database Schema
-- Auto-generated from vehicles.xlsx

-- =========================
-- Users Table
-- =========================
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name VARCHAR(50) NOT NULL UNIQUE,
    password VARCHAR(50) NOT NULL
);

-- =========================
-- Clients Table
-- =========================
CREATE TABLE IF NOT EXISTS clients (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    phone_number VARCHAR(30) NOT NULL UNIQUE,
    chat_summary TEXT,
    last_user_reply TEXT,
    last_bot_reply TEXT,
    last_bot_reply_type VARCHAR(50),
    last_user_message_at DATETIME,
    last_bot_message_at DATETIME,
    info TEXT,
    has_purchased BOOLEAN DEFAULT 0,
    purchase_date DATETIME,
    is_active BOOLEAN DEFAULT 1,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- =========================
-- FollowUp Templates Table
-- =========================
CREATE TABLE IF NOT EXISTS followup_templates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    template_name VARCHAR(100),
    template_body TEXT
);

-- =========================
-- FollowUps Table
-- =========================
CREATE TABLE IF NOT EXISTS followups (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    client_id INTEGER,
    template_id INTEGER,
    scheduled_time DATETIME,
    created_by_employee VARCHAR(100),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (client_id) REFERENCES clients(id),
    FOREIGN KEY (template_id) REFERENCES followup_templates(id)
);

-- =========================
-- Complaints Table
-- =========================
CREATE TABLE IF NOT EXISTS complaints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    client_id INTEGER,
    message_text TEXT,
    is_resolved BOOLEAN DEFAULT 0,
    resolved_at DATETIME,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (client_id) REFERENCES clients(id)
);

-- =========================
-- Motors Table
-- =========================
CREATE TABLE IF NOT EXISTS motors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    english_name VARCHAR(40),
    arabic_name VARCHAR(40),
    company VARCHAR(40),
    agency_name VARCHAR(40),
    moto_type VARCHAR(40),
    price INTEGER,
    engin_capacity VARCHAR(40),
    fule_capacity VARCHAR(40),
    engin_type VARCHAR(40),
    transmission_type VARCHAR(40),
    max_speed VARCHAR(40),
    brake_type VARCHAR(40),
    colors VARCHAR(100),
    notes VARCHAR(100),
    is_available BOOLEAN DEFAULT 1,
    status VARCHAR(40),
    img_url VARCHAR(200)
);

-- =========================
-- Helmets Table
-- =========================
CREATE TABLE IF NOT EXISTS helmets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    english_name VARCHAR(40),
    arabic_name VARCHAR(40),
    company VARCHAR(40),
    price INTEGER,
    helmet_type VARCHAR(40),
    colors VARCHAR(100),
    notes VARCHAR(100),
    is_available BOOLEAN DEFAULT 1,
    status VARCHAR(40),
    img_url VARCHAR(200)
);

-- =========================
-- Motors Data
-- =========================
INSERT INTO motors (english_name, arabic_name, company, agency_name, moto_type, price, engin_capacity, fule_capacity, engin_type, transmission_type, max_speed, brake_type, colors, notes, is_available, status, img_url) VALUES ('SUPPER LIGHT200', 'سوبر لايت', 'كيواى', 'ابو حوا', 'موتوسيكل', 70000, NULL, NULL, 'بنزين', 'يدوي', '110 كم/س', NULL, 'اسود', NULL, 1, 'متاح', 'https://midoalex2025.sirv.com/9.png');
INSERT INTO motors (english_name, arabic_name, company, agency_name, moto_type, price, engin_capacity, fule_capacity, engin_type, transmission_type, max_speed, brake_type, colors, notes, is_available, status, img_url) VALUES ('haojue k4', 'هاوجي k4', 'هاوجى', 'هاوجي', 'موتوسيكل', 58500, '150 cc', '12 لتر', 'بنزين', 'يدوي', '105 كم/س', 'أمامي/خلفي قرص', 'فسفورى', 'محرك قوي – تصميم شبابي', 1, 'متاح', 'https://midoalex2025.sirv.com/21.webp');
INSERT INTO motors (english_name, arabic_name, company, agency_name, moto_type, price, engin_capacity, fule_capacity, engin_type, transmission_type, max_speed, brake_type, colors, notes, is_available, status, img_url) VALUES ('SYM JOYMAX Z+', 'جواى ماكس 300', 'SYM', 'هامر مصر', 'اسكوتر', 215000, '300 cc', '12 لتر', 'بنزين', 'أوتوماتيك', '130 كم/س', 'ABS أمامي وخلفي', 'رصاصى', 'مناسب للسفر الطويل و عليه خصم يوم الخميس فى الكاش 5000', 1, 'متاح', 'https://midoalex2025.sirv.com/25.png');
INSERT INTO motors (english_name, arabic_name, company, agency_name, moto_type, price, engin_capacity, fule_capacity, engin_type, transmission_type, max_speed, brake_type, colors, notes, is_available, status, img_url) VALUES ('SYM JET EVO', 'جيت ايفو', 'SYM', 'هامر مصر', 'اسكوتر', 128000, '150 cc', '6 لتر تقريبًا', 'بنزين', 'أوتوماتيك', '100 كم/س', 'قرص أمامي / خلفي طبلة', 'ابيض', 'رياضي صغير الحجم', 1, 'متاح', 'https://midoalex2025.sirv.com/26.png');
INSERT INTO motors (english_name, arabic_name, company, agency_name, moto_type, price, engin_capacity, fule_capacity, engin_type, transmission_type, max_speed, brake_type, colors, notes, is_available, status, img_url) VALUES ('sym jet 14', 'جيت فور تين', 'SYM', 'هامر مصر', 'اسكوتر', 102000, '150 cc', '6.5 لتر', 'بنزين', 'أوتوماتيك', '100 كم/س', 'باكم أمامي / خلفي طبلة', NULL, 'عائلي عملي', 1, 'متاح', 'https://midoalex2025.sirv.com/27.png');
INSERT INTO motors (english_name, arabic_name, company, agency_name, moto_type, price, engin_capacity, fule_capacity, engin_type, transmission_type, max_speed, brake_type, colors, notes, is_available, status, img_url) VALUES ('sym jet x', 'جيت اكس', 'SYM', 'هامر مصر', 'اسكوتر', 125000, '150 cc', '7.5 لتر', 'بنزين', 'أوتوماتيك', '115 كم/س', 'قرص أمامي وخلفي', 'ازرق', 'تصميم عصري – شاشة رقمية', 1, 'متاح', 'https://midoalex2025.sirv.com/28.png');
INSERT INTO motors (english_name, arabic_name, company, agency_name, moto_type, price, engin_capacity, fule_capacity, engin_type, transmission_type, max_speed, brake_type, colors, notes, is_available, status, img_url) VALUES ('sym sr 200', 'اس ار 200', 'SYM', 'هامر مصر', 'اسكوتر', 85000, '200 cc', '7 لتر', 'بنزين', 'أوتوماتيك', '110 كم/س', 'باكم أمامي وخلفي', NULL, 'قوة أداء جيدة', 1, 'متاح', 'https://midoalex2025.sirv.com/29.jpg');
INSERT INTO motors (english_name, arabic_name, company, agency_name, moto_type, price, engin_capacity, fule_capacity, engin_type, transmission_type, max_speed, brake_type, colors, notes, is_available, status, img_url) VALUES ('sym orbit 150', 'اوربت 150', 'SYM', 'هامر مصر', 'اسكوتر', 60000, '150 cc', '5.5 لتر', 'بنزين', 'أوتوماتيك', '90 كم/س', 'أمامي باكم / خلفي طبلة', 'اوف وايت', 'اقتصادي للمدينة', 1, 'متاح', 'https://midoalex2025.sirv.com/30.jpg');
INSERT INTO motors (english_name, arabic_name, company, agency_name, moto_type, price, engin_capacity, fule_capacity, engin_type, transmission_type, max_speed, brake_type, colors, notes, is_available, status, img_url) VALUES ('sym hasky 200', 'هاسكى', 'SYM', 'هامر مصر', 'اسكوتر', 150000, '200 cc', '12 لتر', 'بنزين', 'اوتوماتيك', '100 كم/س', 'قرص أمامي وخلفي', 'ابيض', 'تصميم كروزر', 1, 'متاح', 'https://midoalex2025.sirv.com/31.jpeg');
INSERT INTO motors (english_name, arabic_name, company, agency_name, moto_type, price, engin_capacity, fule_capacity, engin_type, transmission_type, max_speed, brake_type, colors, notes, is_available, status, img_url) VALUES ('sym hn x 180', 'اتش ان اكس 180', 'SYM', 'هامر مصر', 'موتوسيكل', 98000, '180 cc', '7.5 لتر', 'بنزين', 'أوتوماتيك', '115 كم/س', 'باكم أمامي وخلفي', 'رصاصى', 'رياضي مميز', 1, 'متاح', 'https://midoalex2025.sirv.com/32.webp');
INSERT INTO motors (english_name, arabic_name, company, agency_name, moto_type, price, engin_capacity, fule_capacity, engin_type, transmission_type, max_speed, brake_type, colors, notes, is_available, status, img_url) VALUES ('sym sr 150', 'اس ار 150', 'sym', 'هامر مصر', 'اسكوتر', 80000, '150cc', '7 لتر', 'بنزين', 'أوتوماتيك', '105 كم/س', 'باكم أمامي وخلفي', 'اسود - سماوى', 'مناسب للعمل', 1, 'متاح', 'https://midoalex2025.sirv.com/0064.jpg');
INSERT INTO motors (english_name, arabic_name, company, agency_name, moto_type, price, engin_capacity, fule_capacity, engin_type, transmission_type, max_speed, brake_type, colors, notes, is_available, status, img_url) VALUES ('x-road vigory 150', 'اكس رود 150 فيجورى', 'vigory', 'ابو حوا', 'اسكوتر', 49000, '150cc', '6 لتر', 'بنزين', 'اوتوماتيك', '110 كم/س', NULL, NULL, 'شكل رياضي يخطف العين و ثبات وأمان على الطريق', 1, 'متاح', NULL);
INSERT INTO motors (english_name, arabic_name, company, agency_name, moto_type, price, engin_capacity, fule_capacity, engin_type, transmission_type, max_speed, brake_type, colors, notes, is_available, status, img_url) VALUES ('x-road vigory 200', 'اكس رود 200 فيجورى', 'vigory', 'ابو حوا', 'اسكوتر', 51000, '200cc', '7 لتر', 'بنزين', 'اوتوماتيك', '110 كم/س', NULL, NULL, 'شكل رياضي يخطف العين و ثبات وأمان على الطريق', 1, 'متاح', NULL);
INSERT INTO motors (english_name, arabic_name, company, agency_name, moto_type, price, engin_capacity, fule_capacity, engin_type, transmission_type, max_speed, brake_type, colors, notes, is_available, status, img_url) VALUES ('bee keeway', 'اسكوتر اطفال بي كي واى', 'كى واى', 'ابو حوا', 'اسكوتر', 18000, '450w', 'طارية ليثيوم 48 فولت / 24 أمبير تشيلك مسافة توصل لحد 35 كم بشحنة واحدة', 'كهربائي', 'اوتوماتيك', '40 كم/س', NULL, NULL, 'تصميم عصري، مريح، وموفر للجيب', 1, 'متاح', NULL);
INSERT INTO motors (english_name, arabic_name, company, agency_name, moto_type, price, engin_capacity, fule_capacity, engin_type, transmission_type, max_speed, brake_type, colors, notes, is_available, status, img_url) VALUES ('ket dayun 150', 'كيت دايون 150', 'dayun', 'ابو حوا', 'اسكوتر', 47000, '150cc', '7.5 لتر', 'بنزين', 'اوتوماتيك', '100 كم/س', NULL, NULL, NULL, 1, 'متاح', 'https://midoalex2025.sirv.com/53.jpg');
INSERT INTO motors (english_name, arabic_name, company, agency_name, moto_type, price, engin_capacity, fule_capacity, engin_type, transmission_type, max_speed, brake_type, colors, notes, is_available, status, img_url) VALUES ('400Cc Cruisym', 'كيروزيم 400', 'sym', 'هامر مصر', 'اسكوتر', 298000, '400cc', 'حوالي 14.5 لتر', 'بنزين', 'اوتوماتيك', '166 كم/س', 'ABS لفرامل آمنة، ونظام Advanced Brake Light (ABL) الذي يضيء فرامل الطوارئ تلقائيًا عند الضغط المفاجئ على الفرامل', NULL, 'يوفر قوة 34 حصانًا، مع نظام نقل حركة CVT أوتوماتيكي. يتميز بتصميم أنيق مع إضاءة LED وشاشة TFT ملونة بحجم 7 بوصة، ويوفر أنظمة أمان متقدمة مثل ABS وTCS. تشمل ميزات الراحة نظام Keyless، ومنافذ USB، وتخزين كبير تحت المقعد، مما يجعله مثاليًا للراحة والفخامة.', 1, 'متاح', 'https://midoalex2025.sirv.com/57.webp');
INSERT INTO motors (english_name, arabic_name, company, agency_name, moto_type, price, engin_capacity, fule_capacity, engin_type, transmission_type, max_speed, brake_type, colors, notes, is_available, status, img_url) VALUES ('vlm 200', 'فى ال ام 200cc', 'بينيللى', 'ابو حوا', 'موتوسيكل', 72000, NULL, NULL, 'بنزين', 'يدوي', '115 كم/س', NULL, NULL, NULL, 1, 'متاح', 'https://midoalex2025.sirv.com/10.png');
INSERT INTO motors (english_name, arabic_name, company, agency_name, moto_type, price, engin_capacity, fule_capacity, engin_type, transmission_type, max_speed, brake_type, colors, notes, is_available, status, img_url) VALUES ('x-road keeway 200', 'اكس رود 200', 'keeway', 'ابو حوا', 'اسكوتر', 62000, '200cc', '7 لتر', 'بنزين', 'اوتوماتيك', '110 كم/س', 'فرامل CBS', NULL, 'شبيه وصديق الكيت', 1, 'متاح', 'https://midoalex2025.sirv.com/60.jpg');
INSERT INTO motors (english_name, arabic_name, company, agency_name, moto_type, price, engin_capacity, fule_capacity, engin_type, transmission_type, max_speed, brake_type, colors, notes, is_available, status, img_url) VALUES ('zahra 200', 'زهره 200', 'keeway', 'ابو حوا', 'اسكوتر', 56000, '200cc', '7 لتر', 'بنزين', 'اوتوماتيك', '110 كم/س', 'فرامل CBS', NULL, 'سعر مناسب جدا و شكله شيك', 1, 'متاح', 'https://midoalex2025.sirv.com/61.jpg');
INSERT INTO motors (english_name, arabic_name, company, agency_name, moto_type, price, engin_capacity, fule_capacity, engin_type, transmission_type, max_speed, brake_type, colors, notes, is_available, status, img_url) VALUES ('KEEWAY NEXY 200i', 'نيكسى 200', 'keeway', 'ابو حوا', 'اسكوتر', 90000, '200cc', '7 لتر', 'بنزين', 'اوتوماتيك', '120 كم/س', 'نظام توزيع الفرامل المدمج (CBS)', NULL, 'اصدار جديد و مميز من كى واى', 1, 'متاح', 'https://midoalex2025.sirv.com/62.jpg');
INSERT INTO motors (english_name, arabic_name, company, agency_name, moto_type, price, engin_capacity, fule_capacity, engin_type, transmission_type, max_speed, brake_type, colors, notes, is_available, status, img_url) VALUES ('Keeway Vieste', 'فيستا', 'keeway', 'keeway', 'اسكوتر', 82000, '200cc', NULL, 'بنزين', 'أوتوماتيك', '120 كم/س', 'فرامل CBS', 'كحلى و اسود', 'سعر مناسب جدا و شكله شيك', 1, 'متاح', 'https://ayman2090.sirv.com/ay1.jpg');
INSERT INTO motors (english_name, arabic_name, company, agency_name, moto_type, price, engin_capacity, fule_capacity, engin_type, transmission_type, max_speed, brake_type, colors, notes, is_available, status, img_url) VALUES ('volt way keway', 'فولت واى', 'كيواى', 'ابو حوا', 'اسكوتر', 39000, '2000 واط', '76V / 25Ah', 'كهربائي', 'أوتوماتيك', '70 كم/س', 'أمامي قرص / خلفي CBS', 'جميع الالوان', 'مدى 60 كم / شحن منزلي', 1, 'متاح', 'https://midoalex2025.sirv.com/2.png');
INSERT INTO motors (english_name, arabic_name, company, agency_name, moto_type, price, engin_capacity, fule_capacity, engin_type, transmission_type, max_speed, brake_type, colors, notes, is_available, status, img_url) VALUES ('Symphony ST 200cc', 'سينفوني اس تي ٢٠٠ سي سي', 'sym', 'هامر مصر', 'اسكوتر', 86000, '200cc', '7 لتر', 'بنزين', 'اوتوماتيك', NULL, 'الفرامل: أمامي 260 مم – خلفي 240 مم', 'جميع الالوان', NULL, 1, 'متاح', NULL);
INSERT INTO motors (english_name, arabic_name, company, agency_name, moto_type, price, engin_capacity, fule_capacity, engin_type, transmission_type, max_speed, brake_type, colors, notes, is_available, status, img_url) VALUES ('ket dayun 200', 'كيت دايون 200', 'dayun', 'ابو حوا', 'اسكوتر', 50000, '200cc', '7.5 لتر', 'بنزين', 'اوتوماتيك', '100 كم/س', NULL, 'جميع الالوان', NULL, 1, 'متاح', 'https://midoalex2025.sirv.com/53.jpg');
INSERT INTO motors (english_name, arabic_name, company, agency_name, moto_type, price, engin_capacity, fule_capacity, engin_type, transmission_type, max_speed, brake_type, colors, notes, is_available, status, img_url) VALUES ('Zontes E368', 'زونتوس e368', 'Zontes', 'ابو حوا', 'اسكوتر', 190000, '368cc', '17 لتر', 'بنزين', 'اوتوماتيك', NULL, 'فرامل ABS أمامي و خلفي', 'جميع الالوان', '- نوع العداد : عداد Digital مع شاشة TFT ✅
- نوع الإضائة : إضائة LED بالكامل ✅
- نوع التبريد : تبريد مياه ✅
- سعة تانك البنزين : 17 لتر ✅
- إستهلاك المسافة : 3.5 لتر لكل 100 كم ✅
- نظام حقن الوقود : انجكشن ✅
- مقاس الإطار الأمامي : 15-120/70 ✅
- مقاس الإطار الخلفي : 14-140/70 ✅
- الكاميرات : أمامي و خلفي ✅
- الكماليات : بصمة + مدخل TYPE C & A', 1, 'متاح', 'https://ayman2090.sirv.com/ay2.jpg');
INSERT INTO motors (english_name, arabic_name, company, agency_name, moto_type, price, engin_capacity, fule_capacity, engin_type, transmission_type, max_speed, brake_type, colors, notes, is_available, status, img_url) VALUES ('HAOJIANG NMax 200', 'هوجان ان ماكس 200', 'HAOJIANG', 'ابو حوا', 'اسكوتر', 78000, '200cc', NULL, 'بنزين', 'اوتوماتيك', NULL, 'الفرامل : باكم بنظام CBS', 'جميع الالوان', ' انجين جارد حماية للاسكوتر 💥
تحديد موقع gps map 
يمكن ربطه عن طريق البلوتوث بالموبايل
- كتل مضيئة💥
- شاشة عداد بتقنية TFT💥', 1, 'متاح', NULL);
INSERT INTO motors (english_name, arabic_name, company, agency_name, moto_type, price, engin_capacity, fule_capacity, engin_type, transmission_type, max_speed, brake_type, colors, notes, is_available, status, img_url) VALUES ('dayoun x road 150', 'دايو اكس رود ١٥٠', 'دايون', 'ابوحوا', 'اسكوتر', 46500, '150 cc ', '٨ لتر', 'بنزين', 'اوتوماتيك', NULL, 'الفرامل باكم بنظام cbs', 'جميع الالوان', NULL, 1, 'متاح', NULL);
INSERT INTO motors (english_name, arabic_name, company, agency_name, moto_type, price, engin_capacity, fule_capacity, engin_type, transmission_type, max_speed, brake_type, colors, notes, is_available, status, img_url) VALUES ('dayoun x road 200', 'دايو اكس رود ٢٠٠', 'دايون', 'ابو حوا', 'اسكوتر', 48000, '200cc', '٨ لتر', 'بنزين', 'اوتوماتيك', NULL, 'الفرامل باكم بنظام cbs', 'جميع الالوان', NULL, 1, 'متاح', NULL);
INSERT INTO motors (english_name, arabic_name, company, agency_name, moto_type, price, engin_capacity, fule_capacity, engin_type, transmission_type, max_speed, brake_type, colors, notes, is_available, status, img_url) VALUES ('Tvs HLX plus', 'Tvs هندي', 'Tvs هندي', 'Tvs', 'موتوسيكل', NULL, '100 cc', '١٢ لتر', 'بنزين', 'غيارات', NULL, 'نوع الفرامل : واير أمامي و خلفي ✅', 'جميع الالوان', 'بمحرك ٢٥٠ سي سي بيطلع ٢٥ حصان ، ٤ صبابات و مدعوم بتبريد زيت ، و مساعدين أمامية مقلوبة تحقق أقصي درجات الثبات ', 1, 'متاح', NULL);
INSERT INTO motors (english_name, arabic_name, company, agency_name, moto_type, price, engin_capacity, fule_capacity, engin_type, transmission_type, max_speed, brake_type, colors, notes, is_available, status, img_url) VALUES ('Keeway RKV250', 'كيوي Rkv250', 'Kewway', 'ابوحوا', 'موتوسيكل', NULL, '250 cc', NULL, 'بنزين', 'غيارات', NULL, NULL, NULL, NULL, 1, NULL, NULL);
INSERT INTO motors (english_name, arabic_name, company, agency_name, moto_type, price, engin_capacity, fule_capacity, engin_type, transmission_type, max_speed, brake_type, colors, notes, is_available, status, img_url) VALUES ('Kewway now 50 cc ', 'كيوي ناو ٥٠ سي سي ', 'Kewway', 'ابوحوا ', 'اسكوتر', 31000, '50 cc ', NULL, 'بنزين', 'اوتوماتيك', NULL, NULL, NULL, NULL, 1, NULL, NULL);
INSERT INTO motors (english_name, arabic_name, company, agency_name, moto_type, price, engin_capacity, fule_capacity, engin_type, transmission_type, max_speed, brake_type, colors, notes, is_available, status, img_url) VALUES ('Qj Rko150', 'Qj Rko150', 'Qj', NULL, 'اسكىتر', 70000, '150 cc ', '٧ لتر', 'بنزين', 'اوتوماتيك', NULL, 'باكم امامي وخلفي ', NULL, 'Qj Rko150
سعر البيع ٧٠ الف
سايلنت ستارت
ضمان سنه او ١٢٠٠٠كم 
انجكشن و هايبرد
١٣ حصان
اضاءه ليد
عداد ديجيتال 
ستارت ستوب مود
فرامل باكم امامي و خلفي
تانك ٧ لتر
السرعه القصوي ١٢٥ كم فعليا...', 1, 'متاح', NULL);

-- =========================
-- Helmets Data
-- =========================
INSERT INTO helmets (english_name, arabic_name, company, price, helmet_type, colors, notes, is_available, status, img_url) VALUES ('ls2 hafe', 'ال اس تو نص', 'ls2', 2000, 'نص', 'جميع الالوان', 'عصريه وصلبه و مناسبه للجميع', 1, 'متاح', 'https://midoalex2025.sirv.com/39.png');
INSERT INTO helmets (english_name, arabic_name, company, price, helmet_type, colors, notes, is_available, status, img_url) VALUES ('fibra hafe', 'فيبرا نص', 'fibra', 2000, 'نص', 'جميع الالوان', 'شيك وصلبه و مناسبه للجميع', 1, 'متاح', 'https://midoalex2025.sirv.com/40.png');
INSERT INTO helmets (english_name, arabic_name, company, price, helmet_type, colors, notes, is_available, status, img_url) VALUES ('fibra full', 'فيبرا كامله', 'fibra', 2800, 'كامله', 'جميع الالوان', 'شيك وصلبه و مناسبه للجميع', 1, 'متاح', 'https://midoalex2025.sirv.com/41.png');
INSERT INTO helmets (english_name, arabic_name, company, price, helmet_type, colors, notes, is_available, status, img_url) VALUES ('zeal full', 'زيلا كامله', 'zeal', 3000, 'كامله', 'جميع الالوان', 'جميله وصلبه و مناسبه للجميع', 1, 'متاح', 'https://midoalex2025.sirv.com/42.png');
INSERT INTO helmets (english_name, arabic_name, company, price, helmet_type, colors, notes, is_available, status, img_url) VALUES ('dream full', 'دريم كامله', 'dream', 3000, 'كامله', 'جميع الالوان', 'شيك وصلبه و مناسبه للجميع', 1, 'متاح', 'https://midoalex2025.sirv.com/43.png');
INSERT INTO helmets (english_name, arabic_name, company, price, helmet_type, colors, notes, is_available, status, img_url) VALUES ('aerostat hafe', 'اروستر نص', 'aerostar', 1800, 'نص', 'جميع الالوان', 'شيك وصلبه و مناسبه للجميع', 1, 'متاح', 'https://midoalex2025.sirv.com/44.png');
INSERT INTO helmets (english_name, arabic_name, company, price, helmet_type, colors, notes, is_available, status, img_url) VALUES ('kids hafe', 'خوز اطفال نص', 'kids', 900, 'نص', 'جميع الالوان', 'رائعه للاطفال', 1, 'متاح', NULL);
INSERT INTO helmets (english_name, arabic_name, company, price, helmet_type, colors, notes, is_available, status, img_url) VALUES ('kids full', 'خوز اطفال كامله', 'kids', 1000, 'كامله', 'جميع الالوان', 'رائعه للاطفال', 1, 'متاح', 'https://midoalex2025.sirv.com/46.png');
INSERT INTO helmets (english_name, arabic_name, company, price, helmet_type, colors, notes, is_available, status, img_url) VALUES ('mrc full', 'ام ار سى كامله', 'mrc', 3000, 'كامله', 'جميع الالوان', 'عصريه وقويه و مناسبه للجميع', 1, 'متاح', 'https://midoalex2025.sirv.com/47.webp');
INSERT INTO helmets (english_name, arabic_name, company, price, helmet_type, colors, notes, is_available, status, img_url) VALUES ('spider', 'خوزه اطفال', 'spider', 1200, 'كامله', 'احمر و ازرق', 'مناسبه للاطفال', 1, 'متاح', 'https://midoalex2025.sirv.com/48.jpeg');
