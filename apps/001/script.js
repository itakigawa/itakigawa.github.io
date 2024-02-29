// script.js
document.addEventListener('DOMContentLoaded', () => {
    const fileInput = document.getElementById('fileInput');
    const undoButton = document.getElementById('undoButton');
    const clearButton = document.getElementById('clearButton');
    const showListButton = document.getElementById('showListButton');
    const exportButton = document.getElementById('exportButton');
    const shuffleButton = document.getElementById('shuffleButton');
    const classifyButton = document.getElementById('classifyButton');

    const wordList = document.getElementById('wordList');
    let words = [];
    let removedWords = [];

    fileInput.addEventListener('change', function() {
        const file = this.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                const lines = e.target.result.split('\n');
                // 各単語をオブジェクトとして保存し、selected状態も管理
                words = lines.map(line => ({ text: line.trim(), selected: false })).filter(word => word.text.length > 0);
                renderWords();
            };
            reader.readAsText(file);
        }
    });

    function renderWords() {
        wordList.innerHTML = '';
        words.forEach((wordObj, index) => {
            const wordDiv = document.createElement('div');
            wordDiv.classList.add('tile1');
            wordDiv.textContent = wordObj.text;
            // 選択状態に応じてスタイルを設定
            wordDiv.className = wordObj.selected ? 'tile2' : 'tile1';
            wordDiv.addEventListener('click', () => {
                if (!wordObj.selected) {
                    wordObj.selected = true;
                    wordDiv.className = 'tile2';
                } else {
                    removeWord(index);
                }
            });
            wordList.appendChild(wordDiv);
        });
    }

    function removeWord(index) {
        removedWords.push(words[index]);
        words.splice(index, 1);
        renderWords();
    }

    undoButton.addEventListener('click', () => {
        if (removedWords.length > 0) {
            const lastRemoved = removedWords.pop();
            words.splice(lastRemoved.index, 0, lastRemoved);
            renderWords();
        }
    });

    clearButton.addEventListener('click', () => {
        words = [];
        removedWords = [];
        renderWords();
        fileInput.value = '';
    });

    showListButton.addEventListener('click', () => {
        alert(words.map(wordObj => wordObj.text).join('\n'));
    });

    exportButton.addEventListener('click', () => {
        const blob = new Blob([words.map(wordObj => wordObj.text).join('\n')], { type: 'text/plain' }); // textプロパティを使用
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'exportedWords.txt';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    });

    shuffleButton.addEventListener('click', () => {
        shuffleWords();
        renderWords();
    });

    function shuffleWords() {
        for (let i = words.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [words[i], words[j]] = [words[j], words[i]]; // 要素の交換
        }
    }

    classifyButton.addEventListener('click', () => {
        classifyWords();
        renderWords();
    });

    function classifyWords() {
        const selectedWords = words.filter(wordObj => wordObj.selected);
        const unselectedWords = words.filter(wordObj => !wordObj.selected);
        words = [...selectedWords, ...unselectedWords];
    }
});
