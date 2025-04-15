function Greet(){
    let count = 0
    return new Promise((resolve, reject) => {
        if (count < 5) {
            resolve("Hello")
        }
        else {
            reject("Goodbye")
        }
    })
}
Greet()
.then((message) => {
    console.log(message)
})
.catch((error) => {
    console.log(error)
})