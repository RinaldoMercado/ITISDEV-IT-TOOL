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
    
    // Updated query to match your schema column names
    const query = `
        SELECT 
            u.user_id as id, 
            u.first_name, 
            u.last_name, 
            u.proficiency_level,
            m.module_title as currentModule,
            m.description as currentModuleDesc,
            l.lesson_title as currentLesson,
            l.description as currentLessonDesc,
            (SELECT COUNT(*) FROM quiz_attempts qa WHERE qa.user_id = u.user_id AND qa.passed = 1) as completedQuizzes,
            (SELECT COUNT(*) FROM quiz_attempts qa WHERE qa.user_id = u.user_id) as totalQuizAttempts
        FROM users u
        LEFT JOIN modules m ON u.current_module = m.module_id
        LEFT JOIN lessons l ON u.current_lesson = l.lesson_id
        WHERE u.tutor_id = ? AND u.role = 'student'
    `;

    db.query(query, [tutorId], (err, results) => {
        if (err) {
            console.error("SQL Error:", err.sqlMessage);
            return res.status(500).json({ error: err.sqlMessage });
        }
        res.json(results);
    });
});

app.listen(5000, () => console.log("Server running on port 5000"));