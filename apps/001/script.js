
// script.js
document.addEventListener('DOMContentLoaded', () => {
    const fileInput = document.getElementById('fileInput');
    const undoButton = document.getElementById('undoButton');
    const clearButton = document.getElementById('clearButton');
    const showListButton = document.getElementById('showListButton');
    const exportButton = document.getElementById('exportButton');

    const wordList = document.getElementById('wordList');
    let words = [];
    let removedWords = [];

    fileInput.addEventListener('change', function() {
        const file = this.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                words = e.target.result.split('\n').map(line => line.trim()).filter(line => line.length > 0);
                renderWords();
            };
            reader.readAsText(file);
        }
    });

    function renderWords() {
        console.log('called')
        wordList.innerHTML = '';
        words.forEach((word, index) => {
            const wordDiv = document.createElement('div');
            wordDiv.classList.add('tile');
            wordDiv.textContent = word;
            wordDiv.addEventListener('click', () => {
                removeWord(index);
            });
            wordList.appendChild(wordDiv);
        });
    }

    function removeWord(index) {
        removedWords.push({word: words[index], index});
        words.splice(index, 1);
        renderWords();
    }

    undoButton.addEventListener('click', () => {
        if (removedWords.length > 0) {
            const lastRemoved = removedWords.pop();
            words.splice(lastRemoved.index, 0, lastRemoved.word);
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
        alert(words.join('\n'));
    });

    exportButton.addEventListener('click', () => {
        const blob = new Blob([words.join('\n')], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'exportedWords.txt';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    });

    // ランダムに並び替えるボタンのイベントリスナーを追加
    shuffleButton.addEventListener('click', () => {
        shuffleWords(); // 単語リストをシャッフル
        renderWords(); // 単語リストを再描画
    });

    // 単語リストをシャッフルする関数
    function shuffleWords() {
        for (let i = words.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [words[i], words[j]] = [words[j], words[i]]; // 要素の交換
        }
    }
});
