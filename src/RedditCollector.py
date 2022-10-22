import praw
from praw.models.reddit.subreddit import Subreddit
from praw.models.reddit.submission import Submission
from praw.models.reddit.comment import Comment
from bisect import insort
import pandas as pd
from utilities import Utils


class SortableComment:

    """Class to store a praw.Comment object. Made to sort comments based on number of upvotes.
    """

    def __init__(self, comment: Comment):
        self.comment = comment
        self.upvote = comment.score

    def __str__(self):
        return self.comment.body

    def __eq__(self, other):
        return self.upvote == other.upvote

    def __ne__(self, other):
        return self.upvote != other.upvote

    def __lt__(self, other):
        return self.upvote < other.upvote

    def __gt__(self, other):
        return self.upvote > other.upvote

    def __le__(self, other):
        return self.upvote <= other.upvote

    def __ge__(self, other):
        return self.upvote >= other.upvote


class RedditCollector:

    """Class to manage collecting data from Reddit.
    """

    def __init__(self):

        self.comment_visited = []
        self.comment_queue = []

        self.client_id = 'jGmFCieElW2QvXr2jDDPxw'
        self.secret_token = 'jMmV9AwRsm-mkNusX9exW04qiIneBg'
        self.username = 'brcolli'
        self.password = '204436Lwa'
        self.user_agent = 'Quordata/0.0.1'

        self.reddit = praw.Reddit(client_id=self.client_id,
                                  client_secret=self.secret_token,
                                  username=self.username,
                                  password=self.password,
                                  user_agent=self.user_agent)

    def search_subreddits(self, query: str, limit: int = 5) -> list:
        """Search for subreddits based on a query.

        :param query: The query to search on
        :type query: str
        :param limit: Maximum number of subreddits to get
        :type limit: int

        :return: List of the subreddits from the search
        :rtype: list
        """

        subreddits = self.reddit.subreddits
        return list(subreddits.search(query, limit=limit))

    def search_all_posts(self, query: str, limit: int = 10, time_filter: str = 'week',
                         sort_posts: str = 'relevance') -> list:
        """Do a Reddit-wide search for a query, equivalent to searching on r/all.

        :param query: The query to search for
        :type query: str
        :param limit: Maximum number of posts to get
        :type limit: int
        :param time_filter: Timeframe to get the posts on, default to 'week'
        :type time_filter: str
        :param sort_posts: How to sort the posts, recommended is 'relevance'
        :type sort_posts: str

        :return: List of the post results from the search
        :rtype: list
        """

        rall = self.reddit.subreddit('all')
        return list(rall.search(query, limit=limit, time_filter=time_filter, sort=sort_posts))

    @staticmethod
    def get_hot_posts(subreddit: Subreddit, limit: int = 20) -> list:
        """Get the hot posts in a subreddit.

        :param subreddit: The subreddit to get posts from
        :type subreddit: praw.Subreddit
        :param limit: Maximum number of posts to get
        :type limit: int

        :return: List of the hot posts from the subreddit
        :rtype: list
        """

        return list(subreddit.hot(limit=limit))

    @staticmethod
    def get_top_posts(subreddit: Subreddit, time_filter: str = 'week', limit: int = 20) -> list:
        """Get the top posts in a subreddit.

        :param subreddit: The subreddit to get posts from
        :type subreddit: praw.Subreddit
        :param time_filter: Timeframe to get the posts on, default to 'week'
        :type time_filter: str
        :param limit: Maximum number of posts to get
        :type limit: int

        :return: List of the best posts from the subreddit
        :rtype: list
        """

        return list(subreddit.top(time_filter=time_filter, limit=limit))

    @staticmethod
    def get_top_comments(submission: Submission, limit: int = 10) -> list:
        """Get the top comments on a post.

        :param submission: A post to get comments from
        :type submission: praw.Submission
        :param limit: Maximum number of comments to get
        :type limit: int

        :return: List of the best comments from the post
        :rtype: list
        """

        submission.comment_sort = 'top'
        submission.comment_limit = limit
        return [SortableComment(c) for c in submission.comments.list()]

    def _find_best_replies(self, scomment: SortableComment, limit: int, minimum_upvote: int) -> list:

        """Helper function to find the best replies on a top-level comment. Uses breadth-first search to parse reply
        tree.

        :param scomment: A comment/reply to get replies on
        :type scomment: SortableComment
        :param limit: Maximum number of replies to get
        :type limit: int
        :param minimum_upvote: Minimum number of upvotes a reply should have to be saved
        :type minimum_upvote: int

        :return: List of the best replies from the top-level comment
        :rtype: list
        """

        self.comment_visited.append(scomment)
        self.comment_queue.append(scomment)

        best_comments = []

        # Breadth first search
        while self.comment_queue:

            curr_comment = self.comment_queue.pop(0)

            # Resolve MoreComments instances
            replies = curr_comment.comment.replies
            replies.replace_more(limit=0)

            replies = [SortableComment(r) for r in list(replies)]

            for reply in replies:

                if reply.upvote < minimum_upvote or reply in self.comment_visited:
                    continue

                self.comment_visited.append(reply)
                self.comment_queue.append(reply)

                # If no upvotes or upvotes haven't reached limit or new upvote is larger than smallest saved,
                # then save new upvote
                if not best_comments or len(best_comments) < limit:
                    insort(best_comments, reply)
                elif reply > best_comments[0]:
                    best_comments = best_comments[1:]
                    insort(best_comments, reply)

        return best_comments

    def get_best_replies(self, scomment: SortableComment, limit: int = 5, minimum_upvote: int = 0) -> list:

        """Gets the best replies on a comment. Best is defined as a limited number of comments that have a certain
        number of upvotes. Gives the top number of replies defined by limit.

        :param scomment: The top-level comment to get replies on
        :type scomment: SortableComment
        :param limit: Maximum number of replies to get
        :type limit: int
        :param minimum_upvote: Minimum number of upvotes a reply should have to be saved
        :type minimum_upvote: int

        :return: List of the best replies from the top-level comment
        :rtype: list
        """

        self.comment_queue = []
        self.comment_visited = []

        return self._find_best_replies(scomment, limit=limit, minimum_upvote=minimum_upvote)

    @staticmethod
    def parse_reddit_object(reddit_obj: praw.models.reddit) -> dict:

        """Parses a Reddit object to a dictionary containing the meaningful information.

        :param reddit_obj: Either a Submission or Comment to be parsed to a dictionary
        :type reddit_obj: praw.models.reddit

        :return: Dictionary with Reddit data {Type, Subreddit, Title, Author, Upvotes, Text, #Comments}
        :rtype: dict
        """

        try:
            author = reddit_obj.author.name
        except AttributeError:
            # Handle deleted usernames
            author = '-'

        if type(reddit_obj) == Submission:
            return {'Type': 'Submission', 'Subreddit': reddit_obj.subreddit, 'Title': reddit_obj.title,
                    'Author': author, 'Upvotes': reddit_obj.score, 'Text': reddit_obj.selftext,
                    '# Comments': reddit_obj.num_comments}
        elif type(reddit_obj) == Comment:
            return {'Type': 'Comment', 'Subreddit': reddit_obj.subreddit, 'Title': '-',
                    'Author': author, 'Upvotes': reddit_obj.score, 'Text': reddit_obj.body,
                    '# Comments': len(reddit_obj.replies)}

    def get_reddit_text_from_query(self, query, sort_post_by: str = 'relevance', post_time_filter: str = 'week',
                                   post_limit: int = 10, comment_limit: int = 10, reply_limit: int = 5,
                                   reply_minimum_upvote: int = 10) -> pd.DataFrame:

        """Wrapper to search Reddit for the top posts within a time frame, get the top comments on that post,
        and then get the top replies for each of those comments.

        :param query: String defining the query to search for
        :type query: str
        :param sort_post_by: How to sort the posts, suggested is 'relevance'
        :type sort_post_by: str
        :param post_time_filter: What timeframe to search posts on, default is 'week'
        :type post_time_filter: str
        :param post_limit: Number of posts to request from search
        :type post_limit: int
        :param comment_limit: Number of top-level comments to get on each post
        :type comment_limit: int
        :param reply_limit: Number of replies to get on each top-level comment within each post
        :type reply_limit: int
        :param reply_minimum_upvote: The minimum upvote requirement for replies to be saved
        :type reply_minimum_upvote: int

        :return: Dataframe of all Reddit data collected {Type, Subreddit, Title, Author, Upvotes, Text, #Comments}
        :rtype: pandas.Dataframe
        """

        reddit_data = pd.DataFrame()

        best_posts = self.search_all_posts(query, limit=post_limit, sort_posts=sort_post_by,
                                           time_filter=post_time_filter)

        for post in best_posts:

            reddit_data = pd.concat([reddit_data, pd.DataFrame(self.parse_reddit_object(post), index=[0])],
                                    ignore_index=True)

            best_comments = self.get_top_comments(post, limit=comment_limit)

            for comment in best_comments:

                reddit_data = pd.concat([reddit_data, pd.DataFrame(self.parse_reddit_object(comment.comment),
                                                                   index=[0])],
                                        ignore_index=True)

                best_replies = self.get_best_replies(comment, limit=reply_limit,
                                                     minimum_upvote=reply_minimum_upvote)

                for reply in best_replies:
                    reddit_data = pd.concat([reddit_data, pd.DataFrame(self.parse_reddit_object(reply.comment),
                                                                       index=[0])],
                                            ignore_index=True)

        return reddit_data


if __name__ == '__main__':

    rc = RedditCollector()
    beta_companies = pd.read_csv('../doc/beta_companies.csv')['Name'].tolist()

    for company in beta_companies:

        print(f'Gathering data for {company}')

        data = rc.get_reddit_text_from_query(company)

        Utils.write_dataframe_to_csv(data, f'../data/RedditData/{company}RedditData.csv', write_index=False)
