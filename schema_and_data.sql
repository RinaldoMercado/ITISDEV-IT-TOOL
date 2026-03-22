DROP DATABASE IF EXISTS ManuMano_Db;
CREATE DATABASE ManuMano_Db;
USE ManuMano_Db;

SET FOREIGN_KEY_CHECKS = 0;

CREATE TABLE users (
    user_id INT AUTO_INCREMENT PRIMARY KEY,
    first_name VARCHAR(50),
    last_name VARCHAR(50),
    username VARCHAR(50) UNIQUE,
    email VARCHAR(100) UNIQUE,
    password_hash VARCHAR(255),
    role ENUM('student','tutor','admin'),
    proficiency_level SMALLINT DEFAULT 1,
    contact_number VARCHAR(20),
    bio TEXT,
    profile_picture VARCHAR(255),
    current_module INT DEFAULT 1,
    current_lesson INT DEFAULT 1,
    tutor_id INT DEFAULT NULL,
    FOREIGN KEY (current_module) REFERENCES modules(module_id),
    FOREIGN KEY (current_lesson) REFERENCES lessons(lesson_id),
    FOREIGN KEY (tutor_id) REFERENCES users(user_id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE modules (
    module_id INT AUTO_INCREMENT PRIMARY KEY,
    module_title VARCHAR(100),
    description TEXT,
    module_order INT,
    created_by INT NULL,
    is_custom BOOLEAN DEFAULT FALSE,
    FOREIGN KEY (created_by) REFERENCES users(user_id)
);

CREATE TABLE lessons (
    lesson_id INT AUTO_INCREMENT PRIMARY KEY,
    module_id INT,
    lesson_title VARCHAR(100),
    lesson_order INT,
    description TEXT,
    FOREIGN KEY (module_id) REFERENCES modules(module_id)
);

CREATE TABLE quizzes (
    quiz_id INT AUTO_INCREMENT PRIMARY KEY,
    module_id INT NOT NULL,
    quiz_title VARCHAR(100),
    unlock_after_lesson INT,
    unlocks_module_after_completion INT,
    FOREIGN KEY (module_id) REFERENCES modules(module_id)
);
CREATE TABLE videos (
video_id INT AUTO_INCREMENT PRIMARY KEY,
title VARCHAR(100),
description TEXT,
video_url VARCHAR(255),
thumbnail_url VARCHAR(255),
uploaded_by INT NULL,
video_type ENUM('lesson','quiz','phrasebook','tutor_upload'),
created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

FOREIGN KEY (uploaded_by) REFERENCES users(user_id)
);

CREATE TABLE lesson_videos (
lesson_video_id INT AUTO_INCREMENT PRIMARY KEY,
lesson_id INT NOT NULL,
video_id INT NOT NULL,
video_order INT,

FOREIGN KEY (lesson_id) REFERENCES lessons(lesson_id),
FOREIGN KEY (video_id) REFERENCES videos(video_id)
);


CREATE TABLE quiz_questions (
    question_id INT AUTO_INCREMENT PRIMARY KEY,
    quiz_id INT NOT NULL,
    question_type ENUM(
        'gesture_reenact',
        'gesture_multiple_choice',
        'gesture_text_translation',
        'sign_sentence',
        'multiple_choice',
        'text_input'
    ),
    
    prompt TEXT,
     correct_answer TEXT,
     video_id INT,
    FOREIGN KEY (video_id) REFERENCES videos(video_id),
    
    FOREIGN KEY (quiz_id) REFERENCES quizzes(quiz_id)
);

CREATE TABLE question_options (
    option_id INT AUTO_INCREMENT PRIMARY KEY,
    question_id INT NOT NULL,
    option_text VARCHAR(255),
    is_correct BOOLEAN,
    
    FOREIGN KEY (question_id) REFERENCES quiz_questions(question_id)
);

CREATE TABLE quiz_attempts (
    attempt_id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    quiz_id INT NOT NULL,
    score INT NOT NULL,
    passed BOOLEAN DEFAULT FALSE,
    completed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (user_id) REFERENCES users(user_id),
    FOREIGN KEY (quiz_id) REFERENCES quizzes(quiz_id)
);

CREATE TABLE tutor_assigned_quizzes (
    assignment_id INT AUTO_INCREMENT PRIMARY KEY,
    quiz_id INT NOT NULL,
    student_id INT NOT NULL,
    assigned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    due_date TIMESTAMP NULL,
    completed BOOLEAN DEFAULT FALSE,
    
    FOREIGN KEY (quiz_id) REFERENCES quizzes(quiz_id),
    FOREIGN KEY (student_id) REFERENCES users(user_id)
);

CREATE TABLE phrasebook_categories (
category_id INT AUTO_INCREMENT PRIMARY KEY,
category_name VARCHAR(100)
);

CREATE TABLE phrasebook_entries (
entry_id INT AUTO_INCREMENT PRIMARY KEY,
word VARCHAR(100),
description TEXT,
category_id INT,
video_id INT NOT NULL,

FOREIGN KEY (category_id) REFERENCES phrasebook_categories(category_id),
FOREIGN KEY (video_id) REFERENCES videos(video_id)
);

CREATE TABLE tutor_videos (
tutor_video_id INT AUTO_INCREMENT PRIMARY KEY,
tutor_id INT NOT NULL,
video_id INT NOT NULL,
related_quiz_id INT NULL,
related_lesson_id INT NULL,

FOREIGN KEY (tutor_id) REFERENCES users(user_id),
FOREIGN KEY (video_id) REFERENCES videos(video_id),
FOREIGN KEY (related_quiz_id) REFERENCES quizzes(quiz_id),
FOREIGN KEY (related_lesson_id) REFERENCES lessons(lesson_id)
);

CREATE TABLE gesture_training_data (
data_id INT AUTO_INCREMENT PRIMARY KEY,
video_id INT NOT NULL,
label VARCHAR(100),
uploaded_by INT,
is_verified BOOLEAN DEFAULT FALSE,

FOREIGN KEY (video_id) REFERENCES videos(video_id),
FOREIGN KEY (uploaded_by) REFERENCES users(user_id)
);


USE ManuMano_Db;

-- 1. Wipe everything (Safe way)
SET FOREIGN_KEY_CHECKS = 0;
TRUNCATE TABLE quiz_attempts;
TRUNCATE TABLE quizzes;
TRUNCATE TABLE lessons;
TRUNCATE TABLE modules;
TRUNCATE TABLE users;
SET FOREIGN_KEY_CHECKS = 1;

-- 2. Insert Modules FIRST (The Parents)
INSERT INTO modules (module_id, module_title, description, module_order) VALUES 
(1, 'Foundations of ASL', 'Covers the manual alphabet, numbering 1-20, and the history of Sign Language.', 1),
(2, 'Daily Social Interaction', 'Focuses on common greetings, introducing oneself, and asking basic questions.', 2),
(3, 'Family & Emotions', 'Signs for family members, expressing feelings, and describing relationships.', 3);

-- 3. Insert Lessons SECOND
INSERT INTO lessons (lesson_id, module_id, lesson_title, description, lesson_order) VALUES 
(1, 1, 'The Manual Alphabet', 'Mastering the A-Z hand shapes and transition flow.', 1),
(2, 1, 'Counting 1-10', 'Basic cardinal numbers and palm orientation.', 2),
(3, 2, 'Common Greetings', 'Signs for Hello, Good Morning, and How are you.', 1),
(4, 2, 'Personal Pronouns', 'Using indexing to point to yourself, others, and groups.', 2);

-- 4. Insert Users THIRD (Now the Module IDs 1, 2, and 3 exist!)
INSERT INTO users (user_id, first_name, last_name, role, email, tutor_id, current_module, current_lesson, proficiency_level) VALUES 
(1, 'Aldo', 'Tutor', 'tutor', 'aldo@tutor.com', NULL, NULL, NULL, NULL),
(2, 'Sofia', 'Martinez', 'student', 'sofia@mail.com', 1, 1, 1, 1),
(3, 'Carlos', 'Rivera', 'student', 'carlos@mail.com', 1, 2, 3, 2),
(4, 'Liam', 'Chen', 'student', 'liam@mail.com', 1, 1, 2, 1),
(5, 'Elena', 'Gomez', 'student', 'elena@mail.com', 1, 3, 4, 2);

-- 5. Insert Quizzes and Attempts LAST
INSERT INTO quizzes (quiz_id, module_id, quiz_title) VALUES 
(1, 1, 'Alphabet Mastery Quiz'),
(2, 1, 'Numbers 1-10 Quiz'),
(3, 2, 'Greetings & Etiquette');

INSERT INTO quiz_attempts (user_id, quiz_id, score, passed) VALUES 
-- Sofia: Only 1 attempt, failed. (Shows she might be stuck or unmotivated)
(2, 1, 45, 0), 

-- Carlos: Passed everything first try. (Shows he's ready for harder material)
(3, 1, 95, 1),
(3, 2, 90, 1),

-- Liam: 4 attempts on one quiz! (Shows he is trying hard but struggling with a specific concept)
(4, 1, 30, 0),
(4, 1, 50, 0),
(4, 1, 65, 0),
(4, 1, 85, 1);

SELECT * FROM USERS;

