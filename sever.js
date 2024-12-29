const express = require('express');
const fs = require('fs');
const cors = require('cors');
const app = express();

app.use(cors());

app.get('/api/data', (req, res) => {
    fs.readFile('thông số test.txt', 'utf8', (err, originalData) => {
        if (err) {
            console.error('Failed to read thông số test.txt:', err);
            return res.status(500).json({ error: 'Failed to read thông số test.txt' });
        }

        fs.readFile('AI_output.txt', 'utf8', (err, outputData) => { 
            if (err) {
                console.error('Failed to read AI_output.txt:', err);
                return res.status(500).json({ error: 'Failed to read AI_output.txt' });
            }

            const dataLines = originalData.split('\n').filter(line => line.trim() !== '');
            const outputLines = outputData.split('\n').filter(line => line.trim() !== '');

            const outputMap = {};
            outputLines.forEach(line => {
                const [date, pumpValue] = line.split(' ').map(item => item.trim());
                if (date && pumpValue) {
                    outputMap[date] = pumpValue;
                }
            });

            const result = dataLines.map(line => {
                const parts = line.split(' ').map(item => item.trim());
                if (parts.length < 5) {
                    console.warn(`Invalid data line: ${line}`);
                    return null; // Bỏ qua dòng không hợp lệ
                }
                const date = parts[0];
                const pumpValue = outputMap[date] || 'N/A';

                let pumpLevel = 'Value unrecognized';
                if (pumpValue === '0') {
                    pumpLevel = 'Low';
                } else if (pumpValue === '1') {
                    pumpLevel = 'Medium';
                } else if (pumpValue === '2') {
                    pumpLevel = 'Max';
                }

                return {
                    date: date,
                    soilMoisture: parts[1],
                    airHumidity: parts[2],
                    light: parts[3],
                    temperature: parts[4],
                    pumpLevel: pumpLevel,
                };
            }).filter(item => item !== null);

            res.json(result);
        });
    });
});

app.listen(3000, () => {
    console.log('Server is running on http://localhost:3000');
});