import { Chat } from '../components/Chat'

function Home() {
  return (
      <section className="flex flex-col gap-3">
        <div className="lg:w-3/3">
          <Chat />
        </div>
      </section>
  )
}


export default Home
