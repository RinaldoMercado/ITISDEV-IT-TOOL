const express = require('express');
const mysql = require('mysql2');
const cors = require('cors');
require('dotenv').config();

const app = express();
app.use(cors());
app.use(express.json());

const db = mysql.createConnection({
    host: 'localhost',
    user: 'root',      // Your MySQL user
    password: 'admin', // Your MySQL password
    database: 'ManuMano_Db'
});

app.get('/api/tutees/:tutorId', (req, res) => {
    const { tutorId } = req.params;
    
    // This query joins users with modules and lessons, and counts quiz attempts
    const query = `
        SELECT 
            u.user_id as id, 
            u.first_name, 
            u.last_name, 
            u.proficiency_level,
            m.module_title as currentModule,
            m.module_description as currentModuleDesc,
            l.lesson_title as currentLesson,
            l.lesson_description as currentLessonDesc,
            (SELECT COUNT(*) FROM quiz_attempts qa WHERE qa.user_id = u.user_id AND qa.passed = 1) as completedQuizzes,
            (SELECT COUNT(*) FROM quiz_attempts qa WHERE qa.user_id = u.user_id) as totalQuizAttempts
        FROM users u
        LEFT JOIN modules m ON u.current_module = m.module_id
        LEFT JOIN lessons l ON u.current_lesson = l.lesson_id
        WHERE u.tutor_id = ? AND u.role = 'student'
    `;

    db.query(query, [tutorId], (err, results) => {
        if (err) return res.status(500).json(err);
        res.json(results);
    });
});

app.listen(5000, () => console.log("Server running on port 5000"));