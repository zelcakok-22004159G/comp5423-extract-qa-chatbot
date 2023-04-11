<template lang="pug">
#app
  .container
    .card.base
      .header-row
        label.is-size-4.has-text-weight-semibold Question Answering System
        .header-row
          button.button.is-info(@click="() => this.$refs.fileInput.click()")
            span.icon
              i.fas.fa-file-circle-plus
            span Add Document
      .content
        .placeholder(v-if="!messages.length")
          pre.is-size-4 Ask me some question :)
        .message
          .is-size-6.box(
            :class="{ 'by-user': msg.isSentByUser, 'by-bot': !msg.isSentByUser }",
            v-for="msg in messages"
          ) {{ msg.content }}
        .input-box
          .flex-row
            input.input(placeholder="Press enter to send the question :)", v-model="question", ref="input")
            button.button.is-success(@click="ask", ref="enter")
              span.icon
                i.fas.fa-paper-plane
  form.hidden(ref="form")
    input(type="file", ref="fileInput")
</template>

<script lang="js">
import axios from "axios";

const ENDPOINT = "http://localhost:9000/api"

export default {
  data() {
    return {
      messages: [],
      question: null,
    }
  },
  methods: {
    async ask() {
      const question = this.question;
      this.question = null;
      this.messages.push({
        isSentByUser: true,
        content: question
      })
      try {
        const { data } = await axios.get(`${ENDPOINT}/answer?q=${question}`);
        this.messages.push({
          isSentByUser: false,
          content: data.answer
        })
      } catch(err) {
        this.messages.push({
          isSentByUser: false,
          content: "I can't answer you right now, please wait a minute."
        })
      }
    }
  },
  mounted() {
    const handleFiles = () => {
      const [file] = this.$refs.fileInput.files; /* now you can work with the file list */
      if (!file) {
        return;
      }

      const reader = new FileReader();
      if (file.type !== "text/plain") {
        alert(`Failed to read file ${file.name}, support text/plain format only.`)
      }
      reader.readAsText(file);
      reader.onload = async () => {
        try {
          const { data } = await axios.post(`${ENDPOINT}/documents`, {
            content: reader.result
          })
          this.messages.push({
            isSentByUser: false,
            content: `My database is added a document`
          })
        } catch(err) {
          console.log(err);
          this.messages.push({
            isSentByUser: false,
            content: "I can't update my database, please check."
          })          
        }
      }
      reader.onerror = () => {
        alert("Cannot read file", file.name);
      }
    }

    this.$refs.fileInput.addEventListener("change", handleFiles, false);
    this.$refs.input.addEventListener("keypress", (event) => {
      // If the user presses the "Enter" key on the keyboard
      if (event.key === "Enter") {        
        // Cancel the default action, if needed
        event.preventDefault();
        // Trigger the button element with a click
        this.$refs.enter.click();
      }
    });
  }
};
</script>

<style lang="scss">
@import "../node_modules/bulma/css/bulma.css";
@import url("https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.css");

#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  color: #2c3e50;

  .hidden {
    display: none;
  }

  .container {
    max-width: 900px;

    .card.base {
      margin-left: 8px;
      margin-right: 8px;
      margin-top: 16px;
      height: 95vh;
      padding: 16px;

      .header-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
      }

      .content {
        position: relative;
        height: 83vh;
        margin-top: 32px;

        .placeholder {
          position: absolute;
          width: 100%;
          pre {
            margin-top: 24vh;
            text-align: center;
            color: rgba(128, 128, 128, 0.554);
          }
        }

        .message {
          height: 73vh;
          padding-top: 16px;
          padding-bottom: 50vh;
          overflow-y: auto;

          .is-size-6.box {
            margin-bottom: 12px;
          }

          .by-user {
            background: rgba(266, 231, 94);
            margin-right: 8px;
            margin-left: 32vw;
          }

          .by-bot {
            background: rgba(184, 224, 255);
            margin-left: 8px;
            margin-right: 32vw;
          }
        }

        .input-box {
          .flex-row {
            display: flex;
            flex-direction: row;
            justify-content: flex-start;
            align-items: flex-end;

            .input {
              border-radius: 4px 0 0 4px;
            }

            .button {
              border-radius: 0 4px 4px 0;
            }
          }
        }
      }
    }
  }
}
</style>
